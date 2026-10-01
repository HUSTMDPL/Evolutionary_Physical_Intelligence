# EPI code availability

This package contains simulation code, pretrained base weights and experimental deployment reliance. The digital network generates an SLM1 phase pattern from a scattered image, and the shared SLM2 phase produces the optical reconstruction.

## Files

Keep the Python files in the same directory with their original names.

| File | Purpose |
|---|---|
| `latent space construction.py` | Train the target encoder E and auxiliary image decoder R; also contains the shared model and optical simulator. |
| `pre-train.py` | Jointly train the condition encoder C, denoiser U, phase decoder D and SLM2 parameter. |
| `fine-tune.py` | Adapt the digital parameters to a new paired dataset while keeping the optics fixed. |
| `demo_test.py` | Reconstruct images in simulation and evaluate them against GT when available. |
| `initialization.py` | Configure the devices and acquire paired experimental images. |
| `experiment deployment.py` | Update SLM2 using camera feedback, with the digital network frozen. |
| `experiment transfer train.py` | Adapt the digital network using camera feedback, with SLM2 fixed. |
| `experiment test.py` | Acquire optical reconstructions with the deployed parameters frozen. |
| `latent_space_v13.pt` | Target encoder, auxiliary decoder and latent statistics. |
| `pretrained_v13.pt` | Base weights for simulation, fine-tuning and subsequent deployment. |
| `shared_odr_phase_v13.npy` | Shared SLM2 phase in radians. |

## Base weights

The supplied weights provide a starting point for fine-tuning on a new dataset. They were trained on 11,760 condition–GT pairs: 11,520 static biological pairs and 240 moving biological pairs. Latent construction used 3,120 unique GT file paths. All these data participated in training; scores on the same images measure training fit.

Latent construction completed 80 epochs, with cumulative epoch 79 selected. Joint training completed 22 full epochs and selected epoch 22. These weights have not undergone subsequent fine-tuning or physical deployment.

The demo uses K=8. Other budgets in the manuscript's broader EPI framework require changes to the sampler and configuration; the supplied entries do not expose a K switch.

## Environment and data

Use Python 3.10 or later with PyTorch 2.8, NumPy, Pillow and Matplotlib. CUDA is recommended for training. Experimental serial control also requires `pyserial` and the relevant device SDKs.

Organize paired images as follows. `target` may replace `gt`; filenames and any relative subdirectories should match.

```text
dataset/
  input/
    sample_001.png
  gt/
    sample_001.png
```

Run the commands below from this directory and replace the example paths with local paths.

## Simulation

### Fine-tune and test

Start from `pretrained_v13.pt`, fine-tune on paired images, then test the saved model on a separate evaluation dataset:

```text
python "fine-tune.py" --dataset "ADAPTATION_DATASET" --checkpoint "pretrained_v13.pt" --output "RUNS" --mode C2 --epochs 1
python "demo_test.py" --dataset "EVALUATION_DATASET" --checkpoint "ADAPTED_CHECKPOINT.pt" --output "RUNS"
```

C1 updates U only; C2 updates U and the late part of C. E, R, D, latent statistics and optical parameters stay fixed. Adjust `--epochs` to set the training duration. Each new fine-tuning run starts from a completed pretraining checkpoint; interrupted runs can resume from their own `support/resume_fine_tune_v13.pt`.

To test the base weights directly:

```text
python "demo_test.py" --dataset "EVALUATION_DATASET" --checkpoint "pretrained_v13.pt" --output "RUNS"
```

### Train a new base model

Run latent construction and pretraining on the same dataset:

```text
python "latent space construction.py" --dataset "TRAINING_DATASET" --output "RUNS"
python "pre-train.py" --dataset "TRAINING_DATASET" --latent-checkpoint "LATENT_RUN/latent_space_v13.pt" --output "RUNS"
```

The default durations are 24 latent epochs and 4 joint epochs; set `--epochs` for a longer run. Joint generation starts from independent Gaussian noise. After eight denoising steps, D converts the final latent into SLM1 phase, and the two-SLM simulator forms the image. R is used for latent construction only.

The image loss combines Charbonnier error and `0.1 × (1 − MS-SSIM)`. Joint training adds noise-prediction loss and phase regularization with weight `1e-4`.

### Optical model

Input and output images are RGB at 256 × 256, with an 8 × 32 × 32 latent and T=1000 diffusion levels. The model uses two FSLM-2K73-P02 SLMs separated by 0.50 m, with RGB wavelengths of 635, 532 and 450 nm. The object is at infinity and the image forms in the focal plane of a nominal 200 mm lens. Inter-SLM propagation uses the band-limited angular-spectrum method.

## Experimental deployment

The experimental scripts are interface examples and target Windows, two SLMs connected as extended displays, a Daheng Galaxy monochrome camera, a DCP201B stage controller and a serial RGB source. Device identifiers, phase LUTs, camera calibration and stage positions must be configured for the apparatus.

### Configure and acquire data

Create the configuration file, fill in the device and calibration settings, and check it:

```text
python initialization.py --write-config "SETUP/hardware.json"
python initialization.py --check-config --config "SETUP/hardware.json"
python initialization.py --mock-test
```

### Deploy, transfer and test

Deployment updates SLM2 from camera feedback. Keep the scene static and repeatable during the paired perturbation measurements.

```text
python "experiment deployment.py" --config "SETUP/hardware.json" --dataset "PAIRED_DATASET" --checkpoint "pretrained_v13.pt" --output "EXPERIMENT_RUNS" --steps 20
```

Digital transfer then updates the C1 or C2 parameter subset while retaining the deployed SLM2:

```text
python "experiment transfer train.py" --config "SETUP/hardware.json" --dataset "PAIRED_DATASET" --checkpoint "DEPLOYMENT_RUN/deployed_v13.pt" --output "EXPERIMENT_RUNS" --mode C2 --steps 20
```

Test with a deployed or transferred checkpoint. RGB channels are acquired sequentially.

```text
python "experiment test.py" --config "SETUP/hardware.json" --dataset "EVALUATION_DATASET" --checkpoint "TRANSFER_RUN/transferred_v13.pt" --output "EXPERIMENT_RUNS"
```
