import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --------------------------
# Step 1: Load your product data
# --------------------------
# Make sure your CSV has columns: title, imgurl, producturl, stars, price
products = pd.read_csv('new_products.csv', encoding='latin1')  # or encoding='cp1252'

# Fill missing titles
products['title'] = products['title'].fillna('')

# Ensure stars are numeric
products['stars'] = pd.to_numeric(products['stars'], errors='coerce').fillna(0)

# --------------------------
# Step 2: Convert titles to TF-IDF vectors
# --------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(products['title'])

# --------------------------
# Step 3: Fit NearestNeighbors model
# --------------------------
nn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# --------------------------
# Step 4: Recommendation function
# --------------------------
def recommend_products(search_title, num_recommendations=5):
    # Find product(s) that contain the search title
    matched_products = products[products['title'].str.contains(search_title, case=False, na=False)]

    if matched_products.empty:
        return pd.DataFrame()  # Return empty dataframe if not found

    product_index = matched_products.index[0]

    # Get nearest neighbors
    distances, similar_indices = nn_model.kneighbors(tfidf_matrix[product_index], n_neighbors=num_recommendations + 1)

    similar_indices = similar_indices.flatten()
    similar_indices = [i for i in similar_indices if i != product_index][:num_recommendations]

    recommended_products = products.iloc[similar_indices].sort_values(by='stars', ascending=False)

    return recommended_products[['title', 'imgurl', 'producturl', 'stars', 'price']]

# --------------------------
# Step 5: Streamlit UI
# --------------------------
st.title("Product Recommendation System")
st.write("Type a product title to get 5 similar products!")

search_input = st.text_input("Enter product title:")

if st.button("Recommend"):
    if search_input.strip() == "":
        st.warning("Please enter a product title!")
    else:
        # Always get 5 recommendations
        recommendations = recommend_products(search_input, num_recommendations=5)

        if recommendations.empty:
            st.error("No matching product found!")
        else:
            # Display 5 products side by side
            # Number of columns you want
            num_cols = 3  # You can adjust (5 products -> 3 columns makes 2 products each max)

            # Create columns
            cols = st.columns(num_cols)

            # Split the recommended products into chunks for each column
            for i, col in enumerate(cols):
                # Get the products for this column (2 products per column)
                products_in_col = recommendations.iloc[i * 2: (i + 1) * 2]

                for _, row in products_in_col.iterrows():
                    col.markdown(f"""
                        <div style="
                            width:100%;
                            text-align:center;
                            padding:15px;
                            border-radius:10px;
                            margin-bottom:15px;
                            box-sizing:border-box;
                        ">
                            <h3 style='font-size:16px; margin-bottom:10px; white-space: normal;'>{row['title']}</h3>
                            <img src="{row['imgurl']}" style='width:100%; height:auto; object-fit:contain; margin-bottom:10px;'/>
                            <p style='font-size:14px; margin:0;'><b>Price:</b> ₹{row['price']}</p>
                            <p style='font-size:14px; margin:0;'><b>Rating:</b> {row['stars']}</p>
                            <a href="{row['producturl']}" target="_blank">View Product</a>
                        </div>
                    """, unsafe_allow_html=True)