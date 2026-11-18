"""Actual optimizer-step accounting for the batch-1-accumulation reference run."""
class StepCounter:
    def __init__(self, optimizer):
        self.total = 0
        self.handle = optimizer.register_step_post_hook(self._count)
    def _count(self, optimizer, args, kwargs):
        self.total += 1
    def verify_epoch(self, before, batches, images, expected_batches, expected_images):
        actual = self.total - before
        if batches > 0 and actual == 0:
            raise RuntimeError('Valid batches but zero optimizer updates')
        if actual != batches or batches != expected_batches or images != expected_images:
            raise RuntimeError(f'Epoch accounting mismatch: steps={actual}, batches={batches}/{expected_batches}, images={images}/{expected_images}')
        return actual
