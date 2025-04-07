import os
import json
import boto3
from glob import glob
import pandas as pd
import numpy as np
import psycopg2

"""
【説明】JSON形式で保存したHTMLデータを埋め込み関数でベクトル化する。

【実行方法】実行ファイルと同じ階層で以下のコマンド実行
python3 embed_html_content.py
"""

# 埋め込み関数の設定
DEFAULT_EMBEDDING_CONFIG = {
    "model_id": "cohere.embed-multilingual-v3",
    "chunk_size": 2048,
    "chunk_overlap": 100,
    "enable_partition_pdf": False,
}

# ベクトル重み
title_weight = 1
keyword_weight = 1
dfn_weight = 1.2
index_weight = 1.2
content_weight = 1

# AWS RDSの設定
DB_HOST = "bedrockchatstack-vectorstoreclusterefe7d2f8-xzlsay5mfcdl.cluster-c8n3gy0o0zfk.ap-northeast-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "KjNoMu,TRH1,,9CCp722NzTh=Rrl8=Mj"

# クライアントの設定
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_json_file():
    json_file = glob("./*.json")  # すべてのファイルのパスlist
    if len(json_file) != 1:
        raise ValueError("JSONファイルをひとつにしてください。")
    name = os.path.basename(json_file[0])  # ファイル名のみ取得
    file_name = os.path.splitext(name)[0]  # 拡張子と分離
    return json_file[0], file_name

def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def calculate_document_embeddings(documents: list[str]) -> list[list[float]]:
    def _calculate_document_embeddings(documents: list[str]) -> list[list[float]]:
        payload = json.dumps({"texts": documents, "input_type": "search_document"})
        accept = "application/json"
        content_type = "application/json"

        response = bedrock_client.invoke_model(
            accept=accept, contentType=content_type, body=payload, modelId=DEFAULT_EMBEDDING_CONFIG["model_id"]
        )
        output = json.loads(response.get("body").read())
        embeddings = output.get("embeddings")

        return embeddings

    BATCH_SIZE = 10  # 一度に処理する文書数
    embeddings = []
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        embeddings += _calculate_document_embeddings(batch)

    return embeddings

def embedding(df):
    records = df.to_dict(orient="records")

    for record in records:
        title = record.get('title', '')
        keyword = record.get('keyword', '')
        dfn = record.get('dfn', '')
        index = record.get('index', '')
        content = record.get('content', '')
        
        combined_embedding = np.zeros(1024)
        total_weight = 0
        
        if title:
            title_embedding = calculate_document_embeddings([title])[0]
            combined_embedding += title_weight * np.array(title_embedding)
            total_weight += title_weight

        if keyword:
            keyword_embedding = calculate_document_embeddings([keyword])[0]
            combined_embedding += keyword_weight * np.array(keyword_embedding)
            total_weight += keyword_weight
            
        if dfn:
            dfn_embedding = calculate_document_embeddings([dfn])[0]
            combined_embedding += dfn_weight * np.array(dfn_embedding)
            total_weight += dfn_weight
            
        if index:
            index_embedding = calculate_document_embeddings([index])[0]
            combined_embedding += index_weight * np.array(index_embedding)
            total_weight += index_weight
            
        if content:
            # チャンク分割＞埋め込み計算＞平均
            content_chunks = split_text(content, DEFAULT_EMBEDDING_CONFIG["chunk_size"], DEFAULT_EMBEDDING_CONFIG["chunk_overlap"])
            content_embedding = calculate_document_embeddings(content_chunks)
            content_embedding_ave = [sum(x) / len(content_embedding) for x in zip(*content_embedding)]
            combined_embedding += content_weight * np.array(content_embedding_ave)
            total_weight += content_weight
            
        # 正規化
        combined_embedding /= total_weight
        record['embed_field'] = combined_embedding.tolist()  # npのndarray型(JSONに保存不可)をリストに変換

        """
        # 'embed_field'に重み付けした埋め込み結果を格納
        combined_embedding = (
            title_weight * np.array(title_embedding)
            + keyword_weight * np.array(keyword_embedding)
            + dfn_weight * np.array(dfn_embedding)
            + index_weight * np.array(index_embedding)
            + content_weight * np.array(content_embedding_ave)
        )
        record['embed_field'] = combined_embedding.tolist()  # npのndarray型(JSONに保存不可)をリストに変換
        """
    return pd.DataFrame(records)

def json_to_csv(df):
    csv_path = "data.csv"
    df.to_csv(csv_path, index=False)
    return df

def save_to_local_csv(df, file_name="output.csv"):
    df.to_csv(file_name, index=False)

def save_postgres(table_name, df):
    # データベースに接続
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # テーブルが既に存在する場合に削除
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.commit()

    # テーブル作成
    columns_str = ', '.join([f'"{col}" text' for col in df.columns if col != 'embed_field'])
    query = f"""
        CREATE TABLE "{table_name}" (
            id SERIAL PRIMARY KEY,
            {columns_str},
            embed_field VECTOR(1024)
        );
    """
    cur.execute(query)
    conn.commit()

    # データ挿入
    for i, row in df.iterrows():
        row_values = list(row.values)
        row_values[-1] = "[" + ",".join(map(str, row_values[-1])) + "]"  # embed_fieldをリスト形式に変換
        placeholders = ', '.join(['%s'] * len(row_values))
        insert_query = f'INSERT INTO "{table_name}" ({", ".join(df.columns)}) VALUES ({placeholders})'
        cur.execute(insert_query, row_values)

    conn.commit()
    cur.close()
    conn.close()

def main():
    # JSONデータを取得
    json_path, table_name = get_json_file()
    print("JSONファイルを取得...")
    df = pd.read_json(json_path)
    print("各フィールドの埋め込み計算...")
    df = embedding(df)
    print("CSVへ変換...")
    df = json_to_csv(df)
    #save_to_local_csv(df, "output.csv")
    print("Postgresに保存...")
    save_postgres(table_name, df)
    print("完了!!!")

if __name__ == '__main__':
    main()