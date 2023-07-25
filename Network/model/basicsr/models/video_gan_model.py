from Network.model.basicsr.utils.registry import MODEL_REGISTRY
from Network.model.basicsr.models.srgan_model import SRGANModel
from Network.model.basicsr.models.video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class VideoGANModel(SRGANModel, VideoBaseModel):
    """Video GAN model.

    Use multiple inheritance.
    It will first use the functions of :class:`SRGANModel`:

    - :func:`init_training_settings`
    - :func:`setup_optimizers`
    - :func:`optimize_parameters`
    - :func:`save`

    Then find functions in :class:`VideoBaseModel`.
    """
