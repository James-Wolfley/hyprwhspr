"""Recording cleanup must release status and capture clients before teardown."""

import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_suspend_resume_recovery import _import_main_isolated, patch_app_global


class RecordingCleanupStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, status_path):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._playback_lock = threading.Lock()
        app._playback_session = app._recording_session = None
        app._recording_starting = app._playback_shutdown = False
        app._start_settled = threading.Event(); app._start_settled.set(); app._start_owner = None
        app._autostop_stop_silence_monitor = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._stop_audio_level_monitoring = mock.Mock()
        patcher = patch_app_global(self.main.hyprwhsprApp, 'RECORDING_STATUS_FILE', status_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app

    def test_skipped_audio_discards_realtime_stream(self):
        for audio in (None, [0.0]):
            with self.subTest(audio=audio):
                app = types.SimpleNamespace(
                    _recording_lock=threading.Lock(), is_recording=True,
                    _recording_finalizing=threading.Event(), _recording_session=None,
                    config=mock.Mock(), audio_manager=mock.Mock(),
                    audio_capture=types.SimpleNamespace(stop_recording=lambda: audio,
                                                        lock=threading.Lock(), frames_since_start=1),
                    whisper_manager=mock.Mock(), _is_zero_volume=lambda data: True,
                )
                app.config.get_setting.return_value = 'realtime-ws'
                for name in ('_ptt_reset', '_autostop_stop_silence_monitor', '_clear_mic_osd_preview_text',
                             '_set_visualizer_state', '_stop_audio_level_monitoring', '_write_recording_status',
                             '_restore_recording_playback', '_wait_for_start_settled', '_save_debug_recording',
                             '_notify_zero_volume', '_show_result_and_hide', '_notify_capture'):
                    setattr(app, name, mock.Mock())
                self.main.hyprwhsprApp._stop_recording(app)
                app.whisper_manager.discard_realtime_audio.assert_called_once_with()

    def test_busy_processing_does_not_cancel_owned_transcription(self):
        app = types.SimpleNamespace(_recording_lock=threading.Lock(), is_processing=True,
                                    whisper_manager=mock.Mock())
        self.main.hyprwhsprApp._process_audio(app, [0.5])
        app.whisper_manager.discard_realtime_audio.assert_not_called()

    def test_mute_cancellation_discards_realtime_audio(self):
        app = types.SimpleNamespace(_recording_lock=threading.Lock(), is_recording=True,
                                    _recording_session=None, _recording_finalizing=threading.Event(),
                                    _ptt_reset=mock.Mock(), _cleanup_recording_state=mock.Mock(),
                                    _wait_for_start_settled=mock.Mock(), audio_capture=mock.Mock(),
                                    audio_manager=mock.Mock(), whisper_manager=mock.Mock())
        self.main.hyprwhsprApp._cancel_recording_muted(app)
        app.whisper_manager.discard_realtime_audio.assert_called_once_with()
        self.assertFalse(app._recording_finalizing.is_set())

    def test_phonon_start_waits_for_owned_transcription(self):
        from tests.test_phonon_provider import FakeConfig
        for processing in (True, False):
            app = types.SimpleNamespace(_recording_lock=threading.Lock(), is_recording=False,
                _recording_starting=False, is_processing=processing, _model_initializing=False,
                _backend_init_failed=False, _file_transcription_active=False, _model_operation_active=False,
                _recording_finalizing=threading.Event(), whisper_manager=mock.Mock(_model_manually_unloaded=False),
                config=FakeConfig(transcription_backend='realtime-ws', websocket_provider='phonon'),
                _notify_user=mock.Mock(), _release_blocked_capture=mock.Mock())
            if not processing:
                app._recording_finalizing.set()
            self.main.hyprwhsprApp._start_recording(app)
            app.whisper_manager.realtime_client_missing.assert_not_called()
            self.assertFalse(app.is_recording)
            app._release_blocked_capture.assert_called_once_with()

    def test_blocked_hide_releases_capture_client_and_subscriber_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'recording_status'
            status.write_text('true')
            app = self._app(status)
            server = self.main.RecordingControlServer(
                fifo_path=Path(tmp) / 'recording_control',
                socket_path=Path(tmp) / 'capture.sock',
                on_command=lambda *args: None,
                is_recording=lambda: False,
            )
            app._recording_control_server = server
            stop = threading.Event()
            server._stop_event = stop
            capture_started = threading.Event()
            server._write_fifo = lambda command: capture_started.set() or True
            hide_entered = threading.Event()
            release_hide = threading.Event()
            autostop_stopped = threading.Event()
            errors = []
            clients = []
            workers = []

            def block_hide():
                hide_entered.set()
                release_hide.wait()

            app._autostop_stop_silence_monitor.side_effect = autostop_stopped.set

            def cleanup():
                try:
                    app._cleanup_recording_state()
                except Exception as exc:
                    errors.append(exc)

            def connect_capture():
                capture_started.clear()
                client, connection = socket.socketpair()
                clients.append(client)
                client.settimeout(2)
                worker = threading.Thread(
                    target=server._handle_capture_connection,
                    args=(connection, stop),
                )
                worker.start()
                workers.append(worker)
                client.sendall(b'capture\n')
                self.assertTrue(capture_started.wait(2), 'capture slot was not acquired')
                return client, worker

            app._hide_mic_osd = block_hide
            try:
                client, subscriber = connect_capture()
                cleanup_worker = threading.Thread(target=cleanup)
                cleanup_worker.start()
                workers.append(cleanup_worker)
                self.assertTrue(hide_entered.wait(2), 'cleanup did not reach hide')

                self.assertFalse(status.exists())
                self.assertTrue(
                    autostop_stopped.is_set(),
                    'old auto-stop monitor was not retired before hide blocked',
                )
                self.assertEqual(client.recv(1), b'', 'capture client did not receive EOF')
                subscriber.join(timeout=2)
                self.assertFalse(subscriber.is_alive())
                self.assertFalse(server.has_capture_subscriber())

                # A new client can acquire the slot even while hide is stalled.
                connect_capture()
                self.assertTrue(server.has_capture_subscriber())
            finally:
                release_hide.set()
                stop.set()
                for client in clients:
                    client.close()
                for worker in workers:
                    worker.join(timeout=2)
                self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual(errors, [])
