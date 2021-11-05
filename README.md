# Dissertation

MSc Computer Science Dissertation project

Magnetic Resonance Imaging (MRI) is used in healthcare to obtain detailed images of internal anatomical structures. The images are qualitative rather than quantitative, creating difficulty in comparing findings between patients and between repeat scans taken at different times. Magnetic Resonance Fingerprinting (MRF) is an alternative approach to data acquisition and processing that produces quantitative outputs. It was developed in 2013 and uses an MRI scanner with a novel pseudo random acquisition sequence to elicit unique spatial and temporal signals from different tissue types. Matching quantitative tissue parameters are selected from a dictionary of simulated signals to produce a reconstructed image of the internal anatomical detail. This dictionary matching process takes significant time and is an obstacle that needs to be overcome to expand the utility of MRF scanning.

This dissertation aims to develop a deep learning network that can bypass the dictionary matching process by learning quantitative tissue parameters directly from MRF signals. Four new deep learning approaches are developed that aim to effectively utilise both the spatial and temporal nature of MRF signals. These are compared with dictionary matching as well as another pre-existing state of the art deep learning technique. The work presented here demonstrates that accurate predictions can be made by utilising the temporal information through the use of temporal convolutions and combining this with a perceptual loss to further refine the output.


src directory contains all relevant files, data pre-processing or testing and training code is here with subdirectories containing the models and the dataloaders
