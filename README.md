# Cancer-Image-Analysis-Competition
Repository for the GDSC Cancer Image Analysis Competition.

What I've done: 
- Took a stratified sample of 500 images from the training data, augmented it, and trained a VGG16 image classifier.
  
- Trained a CycleGAN learnt off [this specialization course](https://www.coursera.org/account/accomplishments/specialization/certificate/MK2MTM8QZ9NC) to generate cancerous images from non-cancerous ones (and likewise).
    - The code used can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/AMD-Cloud-Runs/CycleGAN/cycle_gan.py).
    - The resutls produced can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/tree/main/AMD-Cloud-Runs/CycleGAN/CycleGAN_images).
    - The weights for the trained CycleGAN can be found [here](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/tree/main/ckpt/CycleGAN).

- Tried experimenting with multiple options for GNNs, which are explained as follows:
    - **Method 1: Segmentation Using Otsu's Thresholding - [Weights](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/ckpt/GNN/gcn_model-otsu.pth)**
        - **Segmentation Using Otsu's Thresholding**
      
          - Segment foreground objects, specifically nuclei, from the background in grayscale images.
          
        - **Feature Extraction**
      
          - Extract features such as area, perimeter, and eccentricity for each nucleus.
          - These features capture structural and textural aspects of the nuclei
          
        - **Construction of Adjacency Graph**
      
          - Constructing a graph where nodes represent nuclei.
          - Each node in the graph corresponds to a nucleus identified in the image.
          - Edges Connecting Nearby Nuclei: Defining "nearby" based on a distance threshold.

    - **Method 2: Segmentation Using Simple Linear Iterative Clustering (SLIC) - [Weights](https://github.com/shravan-18/Histopathological-Image-Analysis-Competition/blob/main/ckpt/GNN/gcn_model-slic.pth)**
        - **SLIC Segmentation**
            - Images are segmented into superpixels, or segments, based on color similarity.
            - Number of segments, compactness, and sigma -  determine the granularity of the segmentation. 
        
        - **Feature Extraction**
        
            - Mean color of each segment is calculated, serving as a feature vector for that segment (not the ideal choice for this use case, hence did not use this as primary method)
            
        - Construction of Adjacency Graph
        
            - Graph is derived from the shape of the segmented image.
