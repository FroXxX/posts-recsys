from datetime import datetime
from typing import List
from catboost import CatBoostClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import holidays

from TextPreprocessor import TextPreprocessor

def mean_pooling(model_output, attention_mask):
    token_embeddings = torch.stack(model_output["hidden_states"], dim=0)
    token_embeddings = token_embeddings.permute(1,0,2,3)
    input_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 2)
    sum_mask = torch.clamp(input_mask_expanded.sum(2), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return torch.sum(mean_embeddings[:,-4:], dim=1)

def get_embeddings_from_text(text: pd.Series):
    class CustomDataset(Dataset):
        def __init__(self, X, tokenizer):
            self.text = X
            self.tokenizer = tokenizer

        def __len__(self):
            return self.text.shape[0]

        def __getitem__(self, index):
            output = self.text[index]
            output = self.tokenizer(output, return_tensors='pt', padding='max_length', truncation=True)
            return {k: v.reshape(-1) for k, v in output.items()}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    eval_ds = CustomDataset(text.to_numpy(), tokenizer)
    eval_dataloader = DataLoader(eval_ds, batch_size=16)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    embeddings = torch.Tensor().to(device)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = torch.cat([embeddings, mean_pooling(outputs, batch['attention_mask'])])
        embeddings = embeddings.cpu().numpy()
    return embeddings

def get_features_from_text(data_posts):
    logger.info("Preprocessing text.")
    tp = TextPreprocessor(-1)
    data_posts["new_text"] = np.squeeze(tp.transform(data_posts["text"]))
    logger.info("Calculating embeddings.")
    embed_text = get_embeddings_from_text(data_posts["new_text"])
    data_posts = data_posts.drop("new_text", axis=1)
    logger.info("Calculating other features.")
    data_posts["MaxEmbedVal"] = embed_text.max(axis=1)
    data_posts["MeanEmbedVal"] = embed_text.mean(axis=1)
    data_posts["TotalEmbedVal"] = embed_text.sum(axis=1)
    centered = embed_text - embed_text.mean(axis=0)

    pca_processed = PCA(n_components=50).fit_transform(centered)
    cluster_model = KMeans(n_clusters=24, max_iter=1000).fit(pca_processed)
    data_posts["TextCluster"] = cluster_model.labels_
    distances = pd.DataFrame(cluster_model.transform(pca_processed), columns=[ f"DistanceTo{ith}thCluster" for ith in range(1, 25)])
    data_posts = pd.concat((data_posts, distances), axis=1).set_index("post_id")
    return data_posts

def load_data(addr):
    engine = create_engine(addr)
    conn = engine.connect().execution_options(
        stream_results=True
    )
    logger.info("Loading post info.")
    data_posts  = pd.read_sql("SELECT * FROM public.post_text_df", con=conn)
    logger.info("Loading text features.")
    data_posts = get_features_from_text(data_posts)
    logger.info("Loading users info.")
    data_users = pd.read_sql("SELECT * FROM public.user_data", con=conn)
    logger.info("Loading views info.")

    query = """
        SELECT user_id, STRING_AGG(DISTINCT cast(post_id as varchar(5)), ',') as viewed_posts
        FROM public.feed_data
        WHERE action!='like'
        GROUP BY user_id
        """

    data_views = pd.read_sql(query, con=conn)
    data_users = pd.merge(data_users, data_views, on="user_id").set_index("user_id")
    conn.close()
    return data_posts, data_users

def load_model(model_path: str):
    cboost = CatBoostClassifier()
    cboost.load_model(model_path)
    return cboost

def calculate_features_from_timestamp(x: datetime):
    month = x.month
    day_cos = np.cos(2*np.pi*(x.day-1)/31)
    day_sin = np.sin(2*np.pi*(x.day-1)/31)
    hour_cos = np.cos(2*np.pi*x.hour/24)
    hour_sin = np.sin(2*np.pi*x.hour/24)
    day_of_week = x.weekday()
    return [month, day_cos, day_sin, hour_cos, hour_sin, day_of_week]

def is_holiday(dt: datetime, country: str) -> int:
    country_holidays = getattr(holidays, country)()
    return int(dt in country_holidays)
   