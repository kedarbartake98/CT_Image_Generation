# Lung CT image generation by identifying novel semantic masks in PCA space using Deep Reinforcement Learning from user preferences

### Problem Statement:  
  
  - Given Lung CT scan data of a trivial number of patients, synthesize novel Lung CT scans that are physically valid and realistic but not from the original dataset.  
  
### Proposed Approach  
 
#### Latent Space of Semantic Masks (SMs) using PCA:  
   
- We can embed the collection of all SMs into a lower dimensional space via dimensionality reduction. We used Principal Component Analysis (PCA) in our preliminary work. PCA is attractive since it preserves the spatial relationships of the SMs, has a linear inverse transform, and identifies a reduced orthogonal basis that approximates the shape of the SM statistical distribution well. We can then interpolate this PCA space and generate novel SMs which are part of the known SM statistical distribution and have high propensity of being a valid SM. Once the prototypical sample is interpolated; we can transform it back into the original high-dimensional feature space and generate its respective outline.  
  
#### Identifying Novel SMs through Deep Reinforcement Learning:  
  
- RL has shown strong promise for exploring large high-dimensional parameter spaces. RL models learn by a continuous process of receiving rewards and penalties on every action taken and so are able to progressively map out a parameter space into favorable and unfavorable regions and the novel images that reside in them. In order to automatically identify actions that lead to unfavorable configurations we maintain a Reward Classifier that learns from feedback given by an expert user when viewing a synthesized image. This classifier will improve over time progressively, reducing the number of requests for human feedback. 
- Work inspired from [DeepMind Paper](https://arxiv.org/pdf/1706.03741.pdf)

### Dashboard for user input to train Deep RL model  
  
  ![Screenshot](images/Dashboard.png?raw=true)
 
### Description of files:

- PCA_rep.py      :   Class for representing the CT images using PCA. Contains functions for sampling randomly from the PCA plane
- config.py       :   Config file to store data locations and organ details
- graph.py        :   Graph based calculations for the organ splines
- server.py       :   Flask server for the visualization web app
- static/main.js  :   Javascript file for interactions in the web page
- templates/index.html:   Main web page HTML



