**Part 1: Project Description**

Initial Project Scope:

We initially set out to build a model that accurately predicts and forecasts the imminent changes in stock price of Nike based on a combination of quantitative, and qualitative factors, that could potentially be expanded to be used in evaluation for other publicly traded companies. In order to create a more accurate estimation, we wanted to factor in qualitative factors such as the broader macroeconomic conditions of the US economy, competitor and company sentiment, and popularity/trendiness on social media. Additionally, we wanted to output a ‘risk’ factor that would help potential investors gauge the risk of the investment.
Specifically, we wanted to leverage an LSTM neural network that used supervised learning trained on a labeled dataset (historical stock data) to make predictions on new, unseen data (future stock prices). Additionally, we initially projected to feature engineering to gather all relevant, important information. Possible features included in our initial scope: past stock prices, trading volumes, macroeconomic indicators, count of mentions of Nike on Twitter or other social media platforms, count of mentions of Nike on news sources, SEC filings, and sentiment analysis from mentions in news/social media. We wanted to embed all of this information into a vector format that our model could intake to better make its prediction.
In our initial planning, we wanted our LSTM neural network to capture the temporal dependencies in the historical stock data, allowing the system to learn patterns and trends over time. The LSTM would be trained on sequences of historical stock data, where each sequence will represent a specific time period (e.g., a week, a month, etc.) for both the Nike stock, competitors’ stock prices, and the general economy (S&P 500). The LSTM would be designed to take in a sequence of input data (i.e. the features we previously mentioned) and produce a single output value, the predicted future price.

<img width="625" alt="Screen Shot 2023-06-05 at 12 53 27 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/7b06dd47-f4d9-45f2-85e1-88afe14be263">

**Figure 1**: Correlation between calculated Sentiment Score and Nike Stock Price

We wanted a component of this input feature vector to be the result of sentiment analysis from social media and news outlets; we believed the general social sentiment regarding a company could both significantly sway the consumer perception of the firm and be a good metric of its popularity (if the company is trending on social media and popular, market forces would eventually react to this popularity and cause the stock price to rise). We wished to achieve this by utilizing API calls to various social media outlets, namely Twitter and Reddit, as well as new articles.
	Our approach and implementation of sentiment analysis changed significantly throughout the course of our project. Due to the rising popularity of Twitter for widespread discussion, especially regarding the market, we wanted it to be the focal point of our sentiment analysis. We implemented the infrastructure for pulling, cleaning, and interpreting general company (Nike) sentiment on Twitter but due to a change in the social media platform’s management and policies early this year, Twitter discontinued free access to their APIs in February, and significantly raised costs for API calls in February. We were essentially only left with the ability to make 50 read calls per month, a drop in the bucket with such a small sample size for sentiment analysis and our entire implementation (we’ve kept the code for the sake of documentation) was rendered unusable.
	Thus we shifted the scope of our sentiment analysis to other social media platforms, namely Reddit, which also has a prominent presence of retail investors. We evaluated any posts mentioning “Nike” or “NKE” on the most popular investing subreddit and used these posts as input for our sentiment analysis, cleaning our data for preprocessing in order to generate a ‘sentiment score’ that we could then use as another input vector for our LSTM model.  
	Another component of our project that leveraged AI was how we could parse and clean qualitative data (post titles and comments with positive or negative connotation). To this end, we first tokenized each Reddit post and comment into components that were useful to us, namely the words that composed them, the dates they were posted, and their relative popularity (upvotes). We leveraged Vader sentiment analysis which lexically evaluates and quantifies the diction of each word in the reddit post to comprise a ‘sentiment score’ that we were able to model over time and use as a feature vector. This sentiment score was previously seen in Figure 1.
	Finally, we decided to use an LSTM model because of its success with long sequential data problems like stock prediction. More specifically, we wanted to avoid the vanishing gradient problem, while also including information from years prior. This would not be possible with many other models. To do this, we also included a dropout rate to prevent overfitting, and we scaled our data to have a mean of 0 and a standard deviation of 1. This ensured that not one feature was dominating others because of its wide range.

**Part 2: Evaluation**
	With evaluating a LSTM model, we focused on using R2, MAE, and MSE metrics. We chose these because each of them uses different aspects of the model to give predictive analysis and all three of them together can give a comprehensive understanding of the model. The R2 metric measures how much variance in a model is described by the independent variables of the model. In our case, it measures the proportion of stock prices being explained by our features. This was useful in analyzing and comparing models. The MAE and MSE, mean absolute error and mean squared error, respectively, represent the magnitude of the prediction errors made by our model. MAE is the average absolute difference of a predicted point from the actual value, while MSE is the squared difference of a predicted point from the actual value. Through this project we mainly used MAE due to its robustness to outlier data points such as very bad predictions, which could be affected by other variables outside of our model, and thus, we don’t want to heavily penalize our model for these mistakes. After settling on our error metrics, we then wanted to decide which features to include in our model. 
To evaluate the effectiveness of our project and the accuracy of the predictions made, we first wanted to begin by looking at the competitors we were taking in data on to help us with our predictions. Our initial hypothesis was that the stock prices for Adidas, Under Armour, and Skechers would have very high correlation. Thus, we retrieved data for all four companies from the Alpha Vantage API as previously mentioned. The data came in the format of a JSON, and we chose to extract the closing price only from the JSON and insert it into a data frame. Once we had all of the closing data, we modified our data to ensure that the beginning date of the data was synchronized with each other. We then were able to test the first hypothesis of whether there was correlation between the Nike stock and the competitors stock. To do so, first, we graphed the Companies Closing Value ($) vs. Time over the past 12 years.

<img width="649" alt="Screen Shot 2023-06-05 at 12 54 09 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/449cdebe-27fe-46e9-b7dc-59ac329301fb">

**Figure 2**: Shoe Companies Stock Price vs. Time
	From graphing the data, we were able to take away the conclusion that there does exist a very strong correlation between the Nike stock and the Adidas stock, and that their plotted lines overlay in many spots. During the beginning of COVID-19, we can see that all 4 stocks rapidly decline in value and from the middle of 2021 to the start of 2023, the Nike and Adidas stock both declined greatly. We are assigning the reason of this decline to the supply-chain issues that affected many companies. However, the specific drop of Adidas stock in late 2022 was more so attributed to their partnership with rapper Kanye West and their terminated partnership which cost Adidas upwards of 1.2 Billion Euros. This realization also made it more apparent that we could not just use a single competitor, as an external factor unrelated to consumer demand such as comments made by a company partner could attribute to the stock drop, and instead look at multiple competitors at once. 
Further analysis of the graph left us inquiring about the strength of the correlation between the stocks. To answer these questions, we use the Python library Seaborn to visualize a correlation plot that we ran on the dataframe with the stock prices for each competitor.

<img width="605" alt="Screen Shot 2023-06-05 at 12 54 30 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/aab6e5d5-f3e3-4c97-8f90-abeacbeef9d5">

**Figure 3**: Correlation of Nike Stock Price vs. Competitor Stock Prices
	In the top row of the graph, the correlation between Nike and {Nike, Adidas, Under Armour, and Skechers} is shown. Ignoring the Nike-Nike correlation, we can see that the Nike-Adidas and Nike-Skechers correlations are very high, with their values as 0.83 and 0.85 correlation, respectively. However, in steep contrast, there is almost no correlation between the Nike and Under Armour stocks, with a correlation of -0.078. Due to this result, we decided to remove the Under Armour stock information from our evaluation and analysis, as the lack of a positive or negative correlation between the two stocks tells us that the change in Under Armour stock has little to no effect on the Nike stock.
	When looking at the stock data, we used some stock specific metrics to provide further features that more accurately help train a predictive model. To do so, we looked to the Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD), Bollinger Bands, and Momentum. For SMA, we took the average of prices over 7 days and 21 days. MACD is a trend following momentum indicator that shows the relationship between two moving exponential averages (EMAs) which we also calculated. Bollinger bands are two standard deviations both positively and negatively from the SMA. Finally, momentum is the change in the stock over a certain period of time, and in our case, we used from the previous day. 
	To quantify this in terms of our final model and the R2 score and MAE we previously discussed, we changed our feature set to three different varieties. First, one that only includes only Nike data for SMA, EMA, Momentum, and Bollinger Bands. This performed as follows:

<img width="617" alt="Screen Shot 2023-06-05 at 12 55 46 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/bf0a8e52-55c8-4203-8a6d-ca7d41d53c4e">

**Figure 4**: Nike Price Prediction with only Nike data
	Next, we ran a model that included all data from Skechers and Adidas as well as Nike with the same metrics (SMA, EMA, Momentum, and Bollinger Bands). This performed as follows:

<img width="622" alt="Screen Shot 2023-06-05 at 12 56 08 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/a8fe73f9-55ad-49d2-87b3-ff5f4f8b80e7">


**Figure 5**: Nike Price Prediction with Nike, Skechers, and Adidas Data
As you can see, the model was significantly worse. However, given the correlation graphs we previously discussed, we still believed that Adidas and Skechers stock prices and metrics provided key insights that were necessary to use. As such, we created a final feature set that included all of the Nike Data (SMA, EMA, Momentum, and Bollinger Bands), but only the SMA and EMA for Skechers and Adidas. This performed as follows:

<img width="531" alt="Screen Shot 2023-06-05 at 12 56 23 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/204ddc27-79b6-4006-9503-842e0746d776">

**Figure 6**: Nike Price Prediction with Nike Dominant Data with Adidas and Sketchers
Although the R2 score is still the same as with only Nike data, the MAE is lower which is a great sign that this model is stronger. Furthermore, as we ran multiple instances of both over time, we witnessed a trend that the MAE of the model which included Adidas and Skechers data for SMA and EMA was always lower than the MAE for the model with only Nike Data. Thus, we settled on the mixed data as our final feature set. 
	Next, we wanted to look at how different epochs would affect our accuracy and evaluation metrics. To do this, we ran models with 5 epochs, 10 epochs, and 15 epochs. They performed as follows respectively:

<img width="532" alt="Screen Shot 2023-06-05 at 12 56 39 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/fbeaafe1-def2-43b6-adb1-388cd0b8dfe6">

**Figure 7**: Nike Price Prediction after 5 epochs

<img width="627" alt="Screen Shot 2023-06-05 at 12 56 52 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/e51cba2f-1df7-4164-a947-61997cace03f">

**Figure 8**: Nike Price Prediction after 10 epochs

<img width="619" alt="Screen Shot 2023-06-05 at 12 57 05 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/57514c1e-db5a-4bea-b03b-1effbce6daaf">

**Figure 9**: Nike Price Prediction after 15 epochs
Clearly, the model with 5 epochs far surpassed other models with an MAE of 7.97 which is far lower than any other model (including the model with 2 epochs that was previously run). As such, we determined that the 10 and 15 epoch models were overfitting, and that we should stick to 5 epochs.


**Sentiment Analysis**
	We then added our sentiment analysis data to our feature set and retrained the model. The resulting graph is shown here:

<img width="630" alt="Screen Shot 2023-06-05 at 12 57 17 AM" src="https://github.com/roycekoh/lstm_stock/assets/71656996/f0d025d5-8368-40ab-a594-d0087cde5d96">

**Figure 10**: Nike Price Prediction with Sentiment Analysis Included
We saw that the inclusion of our evaluated sentiment analysis score detrimentally affected the predictions of our LSTM model, seen from a lower accuracy and higher mean absolute error score in both 5 and 15 epochs (7.97 absolute loss vs. 16.42 for 5 epochs, 10.58 absolute loss vs. 20.64 for 15 epochs). This can most notably be attributed to our dataset not being large enough (there was on average less than a post a day on Nike), thus not clearly encompassing the public sentiment regarding the company, much less a prediction on how market forces would react. Although the data from incorporating sentiment analysis would agree with our observed predictions using solely financial data that 5 epochs was the perfect middle ground in accuracy and absolute loss, these same results in predictions would conflict with our initial hypothesis that general public sentiment on social media would correlate with stock price.
We gauge this part of the project as a success in that we were able to quantify and generate a realistic representation of public sentiment on social media in a format that was interpretable by our LSTM model, but also a failure in the sense that the lack of relevant data hindered our success. We certainly believe that expanding the scope of media outlets that we look at, varying the types of posts, expanding our sample size, and possibly even factoring in time lags between changes in public sentiment and stock price would greatly help in improving our model.

**Future Steps**
	We believe that incorporating additional macroeconomic data, such as additional S&P 500 behavior, interest rates, and exchange rates would certainly improve the robustness of our model. With so much of the world’s economy and especially Nike’s supply chain relying on global trade, expanding the scope of data sets we look at could certainly be an area of improvement. 
Similarly, looking at posts from additional social media outlets could also allow us to capture a more holistic overview of public sentiment towards the company. Even looking at LinkedIn, expanding the keywords that we search upon, and more carefully pruning for relevant data points (we don’t necessarily have to look at every comment on an arbitrary post) could be room for improvement as well.
Finally, applying the parameters and inputs that we’ve gathered to another model in conjunction with LSTM could yield more accurate results as well. We very preliminarily explored Gradient Boosted Regression Trees as a model that could work for our use case as well due to its ability to better capture non-linear relationships and take into account previous error in training, so expanding our horizons into different models that may be better suited in making our predictions could be another area for improvement we take.

**References**
https://developer.twitter.com/en/docs/twitter-api
https://www.alphavantage.co/documentation/ 
https://www.reddit.com/dev/api/
https://pypi.org/project/vaderSentiment/
https://keras.io/api/layers/recurrent_layers/lstm/
https://keras.io/api/models/sequential/




