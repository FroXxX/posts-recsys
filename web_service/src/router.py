from ast import literal_eval
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Request
import numpy as np
import pandas as pd

from models import PostResponse
from utils import is_holiday
from utils import calculate_features_from_timestamp


router = APIRouter()


@router.get("/post/recommendations/", response_model=List[PostResponse])
def get_recomendation(request: Request, id: int, time: datetime, limit: int = 10):

    app = request.app
    if id not in app.data_users.index:
        raise HTTPException(404, detail="user not found!")

    viewed_posts = list(literal_eval(app.data_users.loc[id, "viewed_posts"]))
    selected_posts = app.data_posts.drop(viewed_posts).drop("text", axis=1)
    user_info = app.data_users.loc[id].drop("viewed_posts").values
    holiday_flag = is_holiday(time, user_info[2])
    user_info = np.concatenate((calculate_features_from_timestamp(time), user_info))

    data = np.zeros(
        (selected_posts.shape[0], user_info.shape[0] + selected_posts.shape[1] + 1),
        dtype=object,
    )
    data[:, : user_info.shape[0]] = user_info
    data[:, user_info.shape[0]: -1] = selected_posts
    data[:, -1] = holiday_flag
    data[:, 0] = data[:, 0].astype(int)  # converting month value to int
    data[:, 5] = data[:, 5].astype(int)  # converting day of week to int
    results = app.cboost.predict_proba(data)[:, 1]
    results = pd.Series(results, index=selected_posts.index.to_numpy())
    results = results.sort_values(ascending=False)
    res_ind = results[:limit].index.to_numpy()
    results = (
        app.data_posts.loc[res_ind, ["topic", "text"]]
        .reset_index()
        .to_dict(orient="records")
    )
    return results
