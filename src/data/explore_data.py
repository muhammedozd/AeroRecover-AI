"""Explore the raw flight dataset."""

from src.data.load_flights import load_flights


df=load_flights()

print("\nVERİ BOYUTU")
print(df.shape)

print("\nSÜTUN İSİMLERİ")
print(df.columns.tolist())

print("\nİLK 5 SATIR")
print(df.head())

print("\nVERİ TİPLERİ VE EKSİK DEĞERLER")
df.info()

print("\nEKSİK DEĞER SAYILARI")
print(df.isna().sum().sort_values(ascending=False))

print("\nHEDEF DEĞİŞKEN DAĞILIMI")
print(df["ARR_DEL15"].value_counts(dropna=False))

print("\nHEDEF DEĞİŞKEN YÜZDELERİ")
print(df["ARR_DEL15"].value_counts(normalize=True, dropna=False) * 100)

print(df[df["ARR_DEL15"]==1].head(1000))