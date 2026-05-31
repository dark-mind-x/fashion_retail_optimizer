import pandas as pd
import numpy as np

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "retail_data.xlsx"

VISIBILITY_MAP = {
    "Eye-Level" : 3,
    "Mid-Level" : 2,
    "Bottom"    : 1
}


def load_products(path=DATA_PATH):

    df = pd.read_excel(path, sheet_name="Product_Data", engine="openpyxl")

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    numeric_columns = [
        "Monthly_Demand",   
        "Unit_Profit",     
        "Unit_Weight_kg",  
        "Product_Width_cm",
        "Product_stacked",
        "Visibility_Score",
        "Trend_Sensitivity",
        "Min_Facing",      
        "Max_Facing",     
        "Current_Facing",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    return df


def load_shelves(path=DATA_PATH):

    df = pd.read_excel(path, sheet_name="Shelf_Rack_Data", engine="openpyxl")

    for col in df.select_dtypes(include=["object","str"]).columns:
        df[col] = df[col].str.strip()

    df["Visibility_Num"] = df["Visibility_Level"].map(VISIBILITY_MAP)

    df["Location_ID"] = df["Rack_ID"] + "-" + df["Shelf_ID"]

    df["Is_Lower"] = df["Shelf_ID"] == "S3"

    df = df.reset_index(drop=True)

    return df


def load_cross_sell(path=DATA_PATH, products=None):

    df = pd.read_excel(path, sheet_name="Cross_Selling_Pairs", engine="openpyxl")

    for col in df.select_dtypes(include=["object","str"]).columns:
        df[col] = df[col].str.strip()
    if products is not None:
        name_to_index = dict(zip(products["Product_Name"], products.index))
        df["Idx_1"] = df["Product_1"].map(name_to_index) 
        df["Idx_2"] = df["Product_2"].map(name_to_index)

        df = df.dropna(subset=["Idx_1", "Idx_2"])
        df[["Idx_1", "Idx_2"]] = df[["Idx_1", "Idx_2"]].astype(int)

    return df


def load_all(path=DATA_PATH):

    products = load_products(path)
    shelves = load_shelves(path)
    cross_sell = load_cross_sell(path, products)


    n = len(products)              
    cross_matrix = np.zeros((n, n))           

    for _, row in cross_sell.iterrows():
        i     = int(row["Idx_1"])
        j     = int(row["Idx_2"])
        score = float(row["Cross_Selling_Score"])
        cross_matrix[i][j] = score  
        cross_matrix[j][i] = score  

    return {
        "products"     : products,
        "shelves"      : shelves,
        "cross_sell"   : cross_sell,
        "cross_matrix" : cross_matrix,
    }


if __name__ == "__main__":

    print("\nLoading data from Excel …\n")
    data = load_all()

    print(f"  PRODUCTS  ({len(data['products'])} rows loaded)")
    cols = ["Product_ID", "Product_Name", "Monthly_Demand",
            "Unit_Profit", "Unit_Weight_kg", "Min_Facing", "Max_Facing"]
    print(data["products"][cols].to_string(index=False))

    print(f"  SHELVES  ({len(data['shelves'])} shelf locations loaded)")
    cols = ["Location_ID", "Shelf Length Limit",
            "Shelf Weight Limit", "Visibility_Num", "Is_Lower"]
    print(data["shelves"][cols].to_string(index=False))

    print(f"  CROSS-SELL PAIRS  ({len(data['cross_sell'])} pairs loaded)")
    cols = ["Product_1", "Product_2", "Cross_Selling_Score"]
    print(data["cross_sell"][cols].to_string(index=False))
    print("  CROSS-SELL MATRIX  (only the paired products shown)")
    mat = data["cross_matrix"]
    pnames = data["products"]["Product_Name"]
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i][j] > 0:
                print(f"  {pnames[i]:25s}  ↔  {pnames[j]:25s}  score = {mat[i][j]:.0f}")

    print("\nData loaded successfully! Ready for the optimizer.\n")
