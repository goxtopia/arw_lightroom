import numpy as np

class BaseProcessor:
    """
    Base class for all image processing filters/nodes.
    Each processor modifies the image and returns the result.
    """
    def process(self, img: np.ndarray, params: dict) -> np.ndarray:
        """
        Processes an image.
        
        Args:
            img: A numpy array representing the image, normalized to float32 [0.0, 1.0].
                 Shape is (H, W, 3) with RGB channels.
            params: A dictionary containing all the filter parameters.
            
        Returns:
            The processed float32 image array normalized to [0.0, 1.0].
        """
        raise NotImplementedError

class ImagePipeline:
    """
    Chains multiple processors together in a sequence.
    Allows modular extension of the post-processing stack.
    """
    def __init__(self, processors: list[BaseProcessor] = None):
        self.processors = processors or []

    def add_processor(self, processor: BaseProcessor):
        self.processors.append(processor)

    def run(self, img: np.ndarray, params: dict) -> np.ndarray:
        """
        Runs the image through the chain of processors.
        """
        out = img
        for processor in self.processors:
            out = processor.process(out, params)
        return out
