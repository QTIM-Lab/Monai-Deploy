# Monai-Deploy
Investigating MONAI Deploy in a series of incrementally more complex examples, ending in a working
pipeline: Orthanc → MONAI Deploy Express → our MAPs → linked DICOM SRs back in Orthanc.

| Folder | What |
|---|---|
| `Simple_Image_Processing/`, `MedNIST_Classifier_App/`, `MedNIST_Classifier_App_Pre-Built_Model/` | App SDK tutorials (below) |
| `AIREADI_Laterality_App/` | laterality model: training code, model weights, tutorial-style MAP |
| [`maps/`](maps/) | **deployable MAPs** and their shared operators |
| [`MONAI_Deploy_Express/`](MONAI_Deploy_Express/LOCAL_SETUP.md) | upstream Express 0.6.0 running the MAPs; our changes in `LOCAL_SETUP.md` |
| `shadow_mode/` | parked: a custom orchestrator explored before Express |

Examples explored:

* [Creating a Simple Image Processing App with MONAI Deploy App SDK](https://monai.readthedocs.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/01_simple_app.html)
* [Deploying a MedNIST Classifier App with MONAI Deploy App SDK](https://monai.readthedocs.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/02_mednist_app.html)
* [Deploying a MedNIST Classifier App with MONAI Deploy App SDK (Prebuilt Model)](https://monai.readthedocs.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/02_mednist_app-prebuilt.html)
* AIREADI Laterality Classifier App with MONAI Deploy App SDK:
    - Trained a ResNet18 on the AI-READI dataset to classify the laterality of colour fundus photographs (left / right eye)
    - Deployed it as a MAP; the production version is `maps/aireadi_laterality`
