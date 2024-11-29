from contextlib import asynccontextmanager
import os
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
import pandas as pd
import uvicorn

from router import router
from utils import load_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    db_connection, model_path = os.getenv("DB_CONNECTION"), os.getenv("MODEL_PATH")
    logger.info("Loading data...")
    if "WEB_LOAD" in os.environ and os.environ["WEB_LOAD"] == "1":
        logger.info(
            "found WEB_LOAD variable set to 1, data will be loaded from database."
        )
        app.data_posts, app.data_users = load_data(db_connection)
    else:
        logger.info("WEB_LOAD is not set, data will be loaded from csv files.")
        app.data_posts = pd.read_csv(
            "data/processed/posts_processed.csv", sep=";"
        ).set_index("post_id")
        app.data_users = pd.read_csv(
            "data/processed/users_processed.csv", sep=";"
        ).set_index("user_id")
    logger.info("loading catboost model...")
    app.cboost = CatBoostClassifier()
    app.cboost.load_model(model_path)
    yield
    del db_connection, model_path, app.data_posts, app.data_users, app.cboost


app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
