import pytest
import model
import torch

G = model.Generator()

def print_generator_shapes(slice_len=32768, latent_dim=100, model_size=64, batch_size=1):
    G = model.Generator(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size)
    z = torch.zeros(batch_size, 1, latent_dim)
    print(f"z shape: {z.shape}")
    x = G.fc(z)
    print(f"fc layer: {x.shape}")
    x = x.view(batch_size, G.dim_mul * G.model_size, 16)
    print(f"reshape: {x.shape}")
    for f in G.convtrans_layers:
        x = f(x)
        print(f"{type(f).__name__}: {x.shape}")

    assert G(z).shape() == torch.Shape(batch_size, 1, slice_len)

def test_generator_output_shape(slice_len=32768, latent_dim=100, model_size=64, batch_size=1):
    G = model.Generator(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size)
    z = torch.zeros(batch_size, 1, latent_dim)
    assert G(z).size() == torch.Size([batch_size, 1, slice_len])

def print_encoder_shapes(slice_len=32768, latent_dim=100, model_size=64, batch_size=1):
    E = model.Encoder(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size)
    x = torch.zeros(batch_size, 1, slice_len)
    print(f"x shape: {x.shape}")
    for f in E.conv_layers:
        x = f(x)
        print(f"{type(f).__name__}: {x.shape}")
    x = x.view(batch_size, 1, 4 * 4 * E.dim_mul * E.model_size)
    print(f"reshape: {x.shape}")
    x = E.fc(x)
    print(f"fc layer: {x.shape}")

    x = torch.zeros(batch_size, 1, slice_len)
    assert E(x).size() == torch.Size([batch_size, 1, latent_dim])

def test_encoder_output_shape(slice_len=32768, latent_dim=100, model_size=64, batch_size=1):
    E = model.Encoder(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size)
    z = torch.zeros(batch_size, 1, slice_len)
    assert E(z).size() == torch.Size([batch_size, 1, latent_dim])

def print_discriminator_shapes(slice_len=32768, latent_dim=100, model_size=64, batch_size=1):
    D = model.Discriminator(slice_len=slice_len, latent_dim=100, model_size=64)
    x = torch.zeros(1, 1, slice_len)
    print(f"x shape: {x.shape}")
    for f in D.x_discrim:
        x = f(x)
        print(f"{type(f).__name__}: {x.shape}")

    z = torch.zeros(1, 1, latent_dim)
    z = z.reshape(1, latent_dim, 1)
    print(f"z shape: {z.shape}")
    for f in D.z_discrim:
        z = f(z)
        print(f"{type(f).__name__}: {z.shape}")

    concat = torch.cat((x, z), dim=1)
    print(f"concat shape: {concat.shape}")
    for f in D.joint_discrim:
        concat = f(concat)
        print(f"{type(f).__name__}: {concat.shape}")

def test_discriminator_output_shape(slice_len=32768,
                                    latent_dim=100,
                                    model_size=64,
                                    discrim_filters=512,
                                    z_discrim_depth=4,
                                    joint_discrim_depth=3,
                                    phaseshuffle_rad=2,
                                    batch_size=1):
    D = model.Discriminator(slice_len=slice_len,
                            latent_dim=latent_dim,
                            model_size=model_size,
                            discrim_filters=discrim_filters,
                            z_discrim_depth=z_discrim_depth,
                            joint_discrim_depth=joint_discrim_depth,
                            phaseshuffle_rad=phaseshuffle_rad)
    x = torch.zeros(batch_size, 1, slice_len)
    z = torch.zeros(batch_size, 1, latent_dim)
    assert D(x, z).size() == torch.Size([1])

if __name__ == "__main__":
    # print("***** GENERATOR *****")
    # for slice_len in [16384, 32768, 65536]:
    #     print(f"----- slice_len = {slice_len} -----")
    #     print_generator_shapes(slice_len=slice_len)
    #
    # print("***** ENCODER *****")
    # for slice_len in [16384, 32768, 65536]:
    #     print(f"----- slice_len = {slice_len} -----")
    #     print_encoder_shapes(slice_len=slice_len)
    # print("***** ENCODER *****")
    # for slice_len in [16384, 32768, 65536]:
    #     print(f"----- slice_len = {slice_len} -----")
    #     print_discriminator_shapes(slice_len)
    test_generator_output_shape()
    test_encoder_output_shape()
    test_discriminator_output_shape()
