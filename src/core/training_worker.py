import threading
from src.core.orchestrator import TrainingOrchestrator


class TrainingWorker:
    """
    Runs training in background thread so Streamlit UI does not block.
    """

    def __init__(self, orchestrator: TrainingOrchestrator):
        self.orchestrator = orchestrator
        self.thread = None
        self.is_running = False
        self.error = None

    def _run(self):
        try:
            self.is_running = True
            self.orchestrator.run()
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_running = False

    def start(self):
        """
        Start training in background.
        """
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def status(self):
        """
        Return current state.
        """
        if self.error:
            return "failed", self.error
        if self.is_running:
            return "running", None
        return "idle", None