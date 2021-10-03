# vocads-challenge

The following repository contains the following files : 

* vocads.ipynb : contains the custom loading class for 20 news dataset along with the fine-tuning pipeline and flask api testing with ngrok
* app.py : flask app to that serves this prediction via a “/predict” route.
* build_embeddings.py : script to generate the embeddings using the model (stored in folder /model) and save them (in folder /embeddings)
* Dockerfile : used to build the docker image 
* requirements.txt : essential packages to install to run the notebook
* requirements_all.txt : all packages of the environment (generated used pip freeze > requirements.txt)
  
