# Super Market Sales Prediction Using LSTM Models

**Project Overview:**
This project focuses on predicting daily sales in a supermarket using LSTM (Long Short-Term Memory) neural networks. LSTMs are well-suited for time series forecasting because they capture temporal dependencies in sequential data.

**Data Visualization (Daily Sales):**
We first explored the sales data to understand trends and seasonality.

<img width="1707" height="615" alt="image" src="https://github.com/user-attachments/assets/43ad8c72-47cd-49cc-ae9c-4422b3b1a0d5" />

**LSTM Model Structure:**
The LSTM network was designed to capture patterns in past sales and predict future values.

<img width="480" height="288" alt="image" src="https://github.com/user-attachments/assets/1e5a7c71-fd36-4a05-9bf5-76f7a1a90ad7" />

**Google Colab Link:**
[Interactive Notebook](https://colab.research.google.com/drive/1eSgm4LwIUCAqpHi6qYHoxGgpl_z9lgYD?usp=sharing)

**Prediction Results:**
We evaluated different models and visualized the predicted vs actual sales.

*Model 0:* <img width="746" height="531" alt="image" src="https://github.com/user-attachments/assets/cfcc6896-bd3e-4f0e-9ffb-df7724c95369" />

*Model 2:* <img width="744" height="526" alt="image" src="https://github.com/user-attachments/assets/3d124acc-c375-4518-a6f4-c9d6ff6fc59f" />

*Model 2 `item-store` predictions:* <img width="716" height="561" alt="image" src="https://github.com/user-attachments/assets/24d3b2ac-586f-4c74-bb59-7ce853854630" />

**Prediction vs Actual Scatter Plot:**
To further assess model accuracy, we plotted predicted vs actual sales. Ideally, points should align along the red dashed line, indicating perfect predictions.

```python
plt.scatter(actual, predicted, alpha=0.3)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual')
plt.show()
```

