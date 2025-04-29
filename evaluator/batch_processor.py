class BatchProcessor:
    def __init__(self, evaluator: LLMEvaluator, batch_size: int = 5):
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.processing_status = "idle"  # idle, processing, completed, error
        self.current_batch = []
        self.error_message = None

    def add_response(self, response: Response) -> None:
        """Add a response to the current batch."""
        self.current_batch.append(response)
        if len(self.current_batch) >= self.batch_size:
            self.process_batch()

    def process_batch(self) -> None:
        """Process the current batch of responses."""
        if not self.current_batch:
            return

        self.processing_status = "processing"
        self.error_message = None

        try:
            for response in self.current_batch:
                evaluation = self.evaluator.evaluate_response(response)
                # Save evaluation to database or storage
                # This is where you'd implement your storage logic
                pass

            self.processing_status = "completed"
        except Exception as e:
            self.processing_status = "error"
            self.error_message = str(e)
            logger.error(f"Error processing batch: {e}")
        finally:
            self.current_batch = []

    def get_status(self) -> Dict[str, Any]:
        """Get the current processing status."""
        return {
            "status": self.processing_status,
            "error_message": self.error_message,
            "batch_size": len(self.current_batch)
        } 