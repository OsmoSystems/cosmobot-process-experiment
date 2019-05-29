import numpy as np

from . import annotate as module


class TestDrawTextOnImage(object):
    def test_output_modified(self):
        test_image = np.zeros((200, 200, 3))
        test_text = 'TEST'

        output_image = module.draw_text_on_image(
            test_image,
            test_text
        )

        assert output_image.any()
        assert not output_image.all()
        # Ensure input is not mutated
        assert (test_image != output_image).any()
