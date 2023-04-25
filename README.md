# AI_AgroWise-Project
[Video Explanation Link]()

[Deployed Project On Cloud](https://agrowise-ai-summative.streamlit.app/)
 - [Introduction](#introduction)
-  [How to run on Cloud](#how-to-run-cloud)
-  [How to run Locally](#how-to-run-locally)


## Introduction
AgroWise is an innovative, AI-powered platform that is set to revolutionize agriculture by providing farmers with intelligent crop recommendations and advanced disease detection capabilities. Building on existing technologies, AgroWise is a sophisticated solution that goes beyond traditional farming practices, offering farmers a smarter, more sustainable way to approach their craft.

## How to run-Cloud

To run the app on the cloud go to this link :  [Deployed Project](https://agrowise-ai-summative.streamlit.app/)

- Crop Recommendation System : The Crop Recommendation system requires you to input the nutrient values of your soil (N-P-K, Rainfall, Ph), as well as your state and city. It's important to note that the N-P-K values entered should be in the form of a ratio. You can refer to a website for further information. Please be aware that when entering your city name, it's best to use commonly recognized names(Kigali) as some remote areas may not be available in the Weather API, which provides humidity and temperature data.

- Plant Disease Detection :  To use the Disease Detection System, you must upload an image of a leaf from your plant. The algorithm will then determine the crop type and whether it is healthy or diseased. If the plant is diseased, the system will identify the cause of the disease and provide suggestions on how to prevent or cure it. Please note that the system currently only supports the following crops.


## How to run locally
- Before the following steps make sure you have virtualenv installed on youur pc and python if not then do  `pip install virtualenv`
- Clone the complete project with git clone `https://github.com/Gladiator07/Harvestify.git` or you can just download the code and unzip it
- Once the project is cloned, open your terminal in the directory where the project was cloned and paste the following block

   ```
   python3 -m venv env
   env/Scripts/Activate.ps1 //In Powershel
   pip install -r requirements.txt 
   ```
- And finally run the project with
    ```python app.py```
    
- Open the localhost url provided after running app.py and now you can use the project locally in your web browser.
