# Nepali Language Auto-Completion Tool using LSTM

## Project Description

This project introduces an LSTM-based Nepali Language Auto-Completion Tool, designed to bridge the gap in Nepali representation within Natural Language Processing (NLP) tools. Utilizing publicly available [dataset](https://nepberta.github.io/) by Sulav Timilsina and Milan Gautam and Binod Bhattarai, 2022, the project aims to navigate the intricate linguistic nuances inherent in the Nepali language. Its core objectives encompass evaluating the performance of different LSTM architectures across various categories of news datasets, subsequently identifying the optimal model for an Editorial category Auto-Completion tool. The overarching goal is to enhance typing efficiency and minimize errors by seamlessly integrating the LSTM model into an intuitive typing tool.

Methodologically, the project emphasizes the utilization of diverse and rigorously curated datasets, ensuring careful evaluation of the model’s performance, and coherent integration into existing tools. By prioritizing these methodologies, the project aims to deliver a robust and effective solution that meets the evolving needs of Nepali language processing. Ultimately, this project serves as both a research endeavor within the dataset made public by the NepBERTa paper and as an initiative in empowering the Nepali language within NLP tools. By merging technical expertise with user-centric solutions, the project strives to facilitate efficient language processing while contributing to the advancement of Nepali representation in the realm of NLP.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- **Python 3.7+**
- **Node.js and npm** (for the Frontend)
- **FastAPI** (for the Backend)
- **Jupyter Notebook or Google Colab** (for running the .ipynb files)
- **Git** (for cloning the repository)

## Installation

### Backend Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Satyasakhadka/nepali-auto-completion-tool.git
    cd nepali-auto-completion-tool/Backend
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
    ```

### Frontend Setup

1. Navigate to the frontend directory:
    ```bash
    cd ../Frontend
    ```

2. Install the required packages:
    ```bash
    npm install
    ```

3. Start the development server:
    ```bash
    npm run dev
    ```
## Usage
Here is the Demo Video of the Usage of Nepali Language Auto Completion Tool:
[![Watch the video]


https://github.com/Satyasakhadka/nepali-auto-completion-tool/assets/83895809/1567af77-a944-4697-aea2-4340c50d9ccd




### Google Colab Setup
1. You can also run the Next_word_predictor.ipynb file in Google Colab.


### Running the Project

1. Ensure the backend server is running:
    ```bash
    cd Backend
    uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
    ```

2. Start the frontend server:
    ```bash
    cd ../Frontend
    npm run dev
    ```

3. Open your web browser and go to `http://localhost:8000` to access the Nepali Language Auto-Completion Tool.

## Contributing

Contributions are always welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.

## Acknowledgments

We would like to express our sincere gratitude towards the Department of Electronics and Computer Engineering, Pulchowk Campus, Institute of Engineering, for their constant guidance and support. Special thanks to our mentors and all the faculty members for their invaluable advice and encouragement.

Authors:
- Satyasa Khadka
- Shuvangi Adhikari
- Sudip Tiwari
