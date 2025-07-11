# **Anomaly Detection in Industrial Control Systems using a Graph Neural Network**

## **Overview**

This project implements a graph-based deep learning model, specifically a **Graph Autoencoder (GAE)**, for anomaly detection on the **HAIEnd 23.05** security dataset. The primary goal is to identify cyber-attacks within a realistic simulation of an industrial boiler process by modeling the system's interconnected components as a graph. This approach leverages not only the time-series data from sensors but also the underlying physical and logical structure of the system to build a more robust detection model.

The final model was trained and optimized to run efficiently on a high-performance **NVIDIA A100 GPU**.

## **Dataset**

This project uses the **HAIEnd 23.05** dataset, which is publicly available from the official HAI repository. This dataset is unique because it provides not only time-series data from sensors but also the internal control logic and physical connection diagrams of the system.

* **Source:** [Official HAI GitHub Repository](https://github.com/icsdataset/hai)

## **Methodology**

The core of this project is to treat the Industrial Control System (ICS) as a graph and use a Graph Neural Network to learn its normal operational behavior.

### **1\. Enriched Graph Construction**

A key challenge with the dataset was that the initial graph, derived only from the control logic files, was very **sparse** (had few connections). This limits a GNN's ability to learn system-wide dependencies. To solve this, we created an **enriched graph**:

* **Logical Connections:** The initial graph was built by parsing the control logic diagrams (dcs\_\*.png) and the merged JSON file to map the flow of information between internal control blocks.  
* **Physical Connections:** Crucially, we added new edges to the graph based on the **physical connections** between components, as shown in the phy\_boiler.png diagram. This included linking components based on the physical flow of water and heat (e.g., connecting a pump's data point to the pressure sensor of the component it feeds).  
* **Name Mapping:** A robust name-mapping logic was developed to link the conceptual node IDs in the graph diagrams (e.g., PIT01) to their corresponding data column names in the CSV files (e.g., DM-PIT01).

### **2\. Model Architecture**

* A **Graph Autoencoder (GAE)** with **Graph Convolutional Network (GCN)** layers was implemented using PyTorch and PyTorch Geometric.  
* The model is **unsupervised**, meaning it learns the system's normal behavior without needing to see attack examples during training.  
* **Encoder:** Two GCN layers process the graph structure and node features (time-series sequences) to create a compressed, low-dimensional latent representation of the entire system's state.  
* **Decoder:** A simple Multi-Layer Perceptron (MLP) reconstructs the original node features from the latent representation.

### **3\. Anomaly Detection**

* The model is trained exclusively on **normal** operational data. Its goal is to learn to reconstruct the system's normal state with the lowest possible error.  
* An attack is detected when the model is given new data and the **reconstruction error** (Mean Squared Error between the input and the reconstructed output) exceeds a statistically determined **threshold**.  
* The threshold was tuned by calculating mean(errors) \+ 3 \* std(errors) on the training data to find an optimal balance between **precision** (avoiding false alarms) and **recall** (finding all attacks).

### **4\. A100 GPU Optimization**

The final script was optimized for maximum performance on a powerful A100 GPU:

* **Tensor Cores (TF32):** Enabled for faster matrix calculations.  
* **Large Batch Size:** Increased to 10000 to maximize parallelism and GPU utilization.  
* **Optimized Data Loading:** Used num\_workers and pin\_memory in the DataLoader to prevent data-loading bottlenecks.  
* **Vectorized Model:** The model's forward pass was rewritten to process entire batches in a single, highly parallel operation, eliminating slow Python loops.

## **How to Set Up and Run**

1. **Prerequisites:** Install the necessary libraries:  
   pip install torch torch\_geometric pandas scikit-learn matplotlib joblib

2. **Dataset:**  
   * Clone the official HAI repository: git clone https://github.com/icsdataset/hai.git  
   * Create a dataset folder in this project's root directory.  
   * Move the haiend-23.05 and graph subdirectories from the cloned HAI repo into your dataset folder.  
3. **Run:** Execute the final Python script (your\_script\_name.py). The script will handle data loading, training, evaluation, and will save the final model, performance report, and plots in the output directory.
