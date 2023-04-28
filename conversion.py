# Model repo: https://github.com/richzhang/colorization
# Conversion ref: https://www.onswiftwings.com/posts/image-colorization-coreml/

import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import torch
from colorizers import *
import matplotlib.pyplot as plt

def convert():
    # load model
    siggraph17_model = siggraph17(pretrained=True).eval()

    # trace model
    example_input = torch.rand(1, 1, 256, 256)
    traced_model = torch.jit.trace(siggraph17_model, example_input)

    # convert to coreml
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input1", shape=(1, 1, 256, 256))]
    )
    coreml_model.save("coremlColorizer.mlmodel")

    # quantize
    coreml_model = ct.models.MLModel('coremlColorizer.mlmodel')
    coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=16)
    coreml_model.save("coremlColorizer.mlmodel")

def test():
    # load
    colorizer_coreml = ct.models.MLModel('coremlColorizer.mlmodel')
    img = load_img('imgs/ansel_adams.jpg')

    # resize, save L channel in both origin and resized resolution
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    # predict a & b channels
    tens_ab_rs = colorizer_coreml.predict({'input1': tens_l_rs.numpy()})['var_518']

    # post-process the output: resize to the original size, concatenate ab with L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_coreml = postprocess_tens(tens_l_orig, torch.from_numpy(tens_ab_rs))

    # save
    plt.imsave('output_siggraph17.png', out_img_coreml)

def main():
    convert()
    test()

if __name__ == "__main__":
   main()
