# DeliverWise Solo Streamlit Demo

This optional demonstration loads the Logistic Regression pipeline created by the verified Kaggle notebook and predicts the probability of late delivery from the same eleven inputs.

The source code avoids new Python-only syntax. The included saved model was
trained with scikit-learn 1.9.1, which is pinned in `requirements.txt`.

## Run locally

1. Open a terminal in this folder.
2. Install the listed packages: `pip install -r requirements.txt`
3. Start the application: `streamlit run app.py`
4. Open the local address shown by Streamlit.

## Run on Streamlit Community Cloud

Upload this folder to a GitHub repository, create an application in Streamlit Community Cloud, and select `app.py` as the entry point. Keep `deliverwise_logistic_pipeline.joblib` in the same folder.

## Model inputs

The form uses item count, seller count, total price, total freight, approval delay, promised days, approval hour, approval weekday, approval month, customer state and seller state.

The included model was created with scikit-learn 1.9.1. The loader also restores
the older `multi_class="auto"` default when an earlier scikit-learn runtime
expects that fitted attribute. For the most reliable deployment, create a new
environment and install `requirements.txt`. If another environment still cannot
load the model, rerun the Kaggle notebook there and replace the joblib file with
the newly exported pipeline.

## Intended use

This interface is an academic demonstration. The model uses historical Brazilian orders from 2016 to 2018 and has low precision at its fixed evaluation threshold. It should not control operational decisions.
