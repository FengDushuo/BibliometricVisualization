# Scientific Research Hot Spot Analysis and Visualization System
The guide can be found in
[Scientific-research-hot-spot-analysis-and-visualization-system.pdf](https://github.com/user-attachments/files/20738252/Scientific-research-hot-spot-analysis-and-visualization-system.pdf)

## Overview

The **Scientific Research Hot Spot Analysis and Visualization System (V1.0)** is designed to uncover and analyze hotspots in scientific research using data visualization and machine learning. The system features a full-stack web framework for the server-side and a client-side developed with HTML, JavaScript, and CSS, using visualization tools like **D3.js** and **Echarts**. Machine learning libraries such as **Scikit-learn** and **Keras** are integrated to perform deep analysis of scientific literature.

## Features

- **Data Visualization**: Visual representation of research trends, including geographical, journal, timeline, and keyword analysis.
- **Interactive Frontend**: An interactive interface for users to explore and visualize scientific data.
- **Machine Learning Integration**: Personalized recommendations based on the analysis of scientific papers and trends.
- **Cross-Platform Support**: Compatible with **Windows** and **Linux** platforms.

## System Components

- **Backend**: Built with a full-stack web framework using asynchronous network libraries.
- **Frontend**: Developed with **HTML**, **JavaScript**, and **CSS**.
- **Data Visualization Tools**: Uses **D3.js** and **Echarts** to visually represent the research data.
- **Machine Learning Libraries**: Incorporates **Scikit-learn** and **Keras** for advanced analysis of scientific literature.

## Installation

### Backend Setup

1. Deploy the server application to your local environment.
2. After the server starts, you can access the system via the following:
   - **Local**: `http://127.0.0.1:8000`
   - **Internal Network**: `http://<internal_IP>:8000`
   - **External Network**: `http://<external_IP>:8000`

### Running the Software Package

1. Download and unzip the software package.
2. Run the `server.exe` to start the server on your machine.
3. Access the system using the provided URLs.

## Usage Instructions

### Register and Log In

1. **Registration**: New users must click the registration button on the home page to provide a username, email, and password.
2. **Login**: Existing users can log in using their credentials.
3. **Password Recovery**: Users who forget their password can reset it through the "Forgot Password" feature.

### Data Graph Visualization

1. Download data from the **Web of Science** retrieval platform to analyze research hotspots.
2. Use the **Data Graph Visualization** button to enter the visualization module and perform operations.

#### Modules Available:

- **Regional Analysis**: Represents the number of documents from various regions, showing the geographical distribution of research.
- **Journal Analysis**: Analyzes publications across different journals.
- **Timeline Analysis**: Visualizes trends in publication over time.
- **Keyword Analysis**: Uses machine learning algorithms to extract and display research keywords.

### Upload Data

1. Users can upload musculoskeletal ultrasound image data for analysis. The system uses the **U-Net architecture** for recognizing aponeurosis and muscle bundles.
2. Upload data through the "Upload Data" module and follow the instructions for selecting and visualizing the data.

### Literature Recommendations

1. **Personalized Recommendations**: The system recommends literature based on user interests, using machine learning algorithms to predict publication trends and identify potentially highly-cited documents.
2. **Access Recommendations**:
   - **From Data Graph Visualization**: Click the recommendation button.
   - **From the Home Page**: Navigate to the recommendation page and select the relevant publication characteristics to get recommendations.

## Contributing

We welcome contributions to improve the system. To contribute:

1. Fork the repository.
2. Clone your fork and create a branch.
3. Make your changes and submit a pull request.

## License

This project is licensed under the MIT License.

