from abc import ABC, abstractmethod

class VQAInterface(ABC):
    @abstractmethod
    def __call__(self, images, query_batch, *token_groups_of_interest):
        pass

    @abstractmethod
    def generate_system_cache_with_images(self, images):
        pass

    @abstractmethod
    def clear_system_cache(self):
        pass