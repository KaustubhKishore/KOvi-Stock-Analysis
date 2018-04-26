import tkinter
from tkinter import ttk
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas_datareader.data as web
import bs4 as bs
import pickle
import requests
from tkinter import messagebox
from datetime import timedelta
from matplotlib import style
import time
import numpy as np
import tkinter.messagebox as box
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


style.use('ggplot')
fontsets = ("Helvetica", 12)


class KOvi(tkinter.Tk):
    def __init__(self):
        tkinter.Tk.__init__(self)
        cont = tkinter.Frame(self)
        tkinter.Tk.wm_title(self, "KOvi Stock Analysis")

        cont.pack(side="top", fill="both", expand=True)
        cont.grid_rowconfigure(0, weight=1)
        cont.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, HomePage):
            frame = F(cont, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont1):
        frame = self.frames[cont1]
        frame.tkraise()


class StartPage(tkinter.Frame):

    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        l1 = tkinter.Label(self, text="Project KOvi", font=fontsets)
        l1.pack(padx=10, pady=10)

        def CheckLogin():
            username = entUserName.get()
            password = entPssword.get()
            if username == 'admin' and password == 'admin':
                box.showinfo('info', 'Welcome Kaustubh Kishore')
                controller.show_frame(HomePage)
            elif username == 'user' and password == 'password':
                box.showinfo('info', 'Welcome Ouvais Saifi')
                controller.show_frame(HomePage)
            else:
                box.showinfo('info', 'Invalid Login')

        lblUserName = ttk.Label(self, text = "Username: ", font = fontsets)
        lblUserName.pack()

        entUserName = ttk.Entry(self)
        entUserName.pack()

        lblPassword  = ttk.Label(self, text = 'Password: ', font = fontsets)
        lblPassword.pack(padx = 15, pady = 5)

        entPssword = ttk.Entry(self, show = "*")
        entPssword.pack(padx = 15, pady = 5)

        Button_Login = ttk.Button(self, text = 'Check Login', command = CheckLogin)
        Button_Login.pack(padx = 5)


class HomePage(tkinter.Frame):

    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        self.tickers = []
        try:
            with open("SP500tickers.pickle", "rb") as f:
                self.tickers = pickle.load(f)

        except:
            self.getSP500()

        l1 = tkinter.Label(self, text = "Home Page", font= fontsets)

        btnHome = ttk.Button(self, text = "Logout", command = lambda: controller.show_frame(StartPage))

        btnGetData = ttk.Button(self, text = "Google Data", command = self.GetData)

        btnGetTickers = ttk.Button(self, text = "Tickers", command = self.getSP500)

        btnCombineData = ttk.Button(self, text = "Compile Data", command = self.compile_data)

        btnMorningstar = ttk.Button(self, text = "Morningstar Data", command = self.temp_GetData)

        btnCorTable = ttk.Button(self, text = "Correlation Table", command = self.visualize_data )

        btnShowTickerList = ttk.Button(self, text = "Ticker List", command = self.ticker_list)

        lblBtnGetData = ttk.Label(self, text = "Retrieve Data from Google API: " , justify = 'right')
        lblBtnGetTickers = ttk.Label(self, text = "Update S&P 500 List: ")
        lblBtnCombineData = ttk.Label(self, text = "Combine Data: ")
        lblBtnMorningstar = ttk.Label(self, text = "Retrieve Data from Morningstar API: ")
        lblBtnCorTable = ttk.Label(self, text = "Create Correlation Table: ")

        l1.grid(row = 0, pady = 20, columnspan = 3)

        lblBtnGetTickers.grid(row = 1)
        btnGetTickers.grid(row = 1, column = 1)

        lblBtnGetData.grid(row = 2)
        btnGetData.grid(row = 2, column = 1)

        lblBtnMorningstar.grid(row = 3)
        btnMorningstar.grid(row = 3, column = 1)

        lblBtnCombineData.grid(row = 4)
        btnCombineData.grid(row = 4, column = 1)

        lblBtnCorTable.grid(row = 5)
        btnCorTable.grid(row = 5, column = 1)




        def process_data_for_labels(ticker):
            hm_days = 7
            df = pd.read_csv('sp500_closes.csv', index_col = 0)
            ticks = df.columns.values.tolist()
            df.fillna(0, inplace = True)

            for i in range(1, hm_days + 1):
                df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker] ) / df[ticker]

            df.fillna(0, inplace = True)
            return ticks, df


        def buy_sell_hold(*args):
            cols = [c for c in args]
            requirement = 0.02
            for col in cols:
                if col > requirement:
                    return 1
                if col < -requirement:
                    return -1

            return 0

        def extract_featuresets(ticker1):
            tickers, df = process_data_for_labels(ticker1)

            df['{}_target'.format(ticker1)] = list(map(buy_sell_hold,
                                                       df['{}_1d'.format(ticker1)],
                                                       df['{}_2d'.format(ticker1)],
                                                       df['{}_3d'.format(ticker1)],
                                                       df['{}_4d'.format(ticker1)],
                                                       df['{}_5d'.format(ticker1)],
                                                       df['{}_6d'.format(ticker1)],
                                                       df['{}_7d'.format(ticker1)]
                                                       ))
            vals = df['{}_target'.format(ticker1)].values.tolist()
            str_vals = [str(i) for i in vals]
            print('Data Spread: ', Counter(str_vals))

            box.showerror("Spread", Counter(str_vals) )

            df.fillna(0, inplace = True)
            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace = True)

            df_vals = df[[ticker for ticker in tickers]].pct_change()
            df_vals = df_vals.replace([np.inf, -np.inf], 0)
            df_vals.fillna(0, inplace = True)

            X = df_vals.values
            y = df['{}_target'.format(ticker1)].values

            return X, y, df

        def do_ml():
            tick = entPredict.get()
            X, y, df = extract_featuresets(tick)

            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.30)


            clf = VotingClassifier([
                ('lsvc', svm.LinearSVC()),
                ('knn', neighbors.KNeighborsClassifier()),
                ('rfor', RandomForestClassifier())
            ])

            clf.fit(X_train, y_train)
            confidence = clf.score(X_test, y_test)
            predictions = clf.predict(X_test)

            toor = tkinter.Tk()

            lblConfidence = ttk.Label(toor, text = confidence)
            lblPredictions = ttk.Label(toor, text = Counter(predictions))
            lblConfidence1 = ttk.Label(toor, text = "Accuracy: ")
            lblPredictions1 = ttk.Label(toor, text = "Buy Sell Or Hold (1, -1, 0) : ")

            lblConfidence1.grid(row = 0)
            lblConfidence.grid(row = 0, column = 1, padx = 5)
            lblPredictions1.grid(row = 1)
            lblPredictions.grid(row = 1, column = 1, padx = 5)
            toor.mainloop()


            return confidence


        def live_graph():

            ticker = entTicker.get()

            if os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
                plt.plot(df['Date'], df['Close'])
                plt.xlabel('Date')
                plt.ylabel('Stock Price (USD)')
                plt.show()
            else:
                box.showerror("Error", "No data for given Ticker found")


        lblInputTicker = ttk.Label(self, text = "Enter Ticker to get Price Graph: ")
        entTicker = ttk.Entry(self)
        btnSubmitTicker = ttk.Button(self, text = "Submit", command = live_graph)

        lblInputTicker.grid(row = 6)
        entTicker.grid(row = 6, column = 1)
        btnSubmitTicker.grid(row = 6, column = 2, padx = 10)

        btnHome.grid(row = 9, pady = 20, columnspan = 3)
        btnShowTickerList.grid(row = 8, column = 1, pady = 5)

        lblPredict = ttk.Label(self, text = "Make prediction for: ")
        entPredict = ttk.Entry(self)

        lblPredict.grid(row = 7)
        entPredict.grid(row = 7, column = 1)

        btnSubmit2 = ttk.Button(self, text = "Predict", command = do_ml)
        btnSubmit2.grid(row = 7 ,column = 2)

    def getSP500(self):
        if (os.path.exists('SP500tickers.pickle') == False):
            resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                self.tickers.append(ticker)
            with open("SP500tickers.pickle", "wb") as f:
                pickle.dump(self.tickers, f)

    def GetData(self):
        start = dt.datetime(2017, 1, 1)
        end = dt.datetime.now() - timedelta(days = 1)
        print(self.tickers)
        if len(self.tickers) == 0:
            messagebox.showinfo("Ticker List Underflow", "Retrieve latest S&P500 list using 'Get Tickers' button")
        else:
            if not os.path.exists('stock_dfs'):
                os.makedirs('stock_dfs')

            for ticker in self.tickers:
                if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                    df = web.DataReader(ticker, 'google', start, end)
                    df.to_csv('stock_dfs/{}.csv'.format(ticker))
                    time.sleep(1)

                else:
                    print('Data for {} already exists'.format(ticker))

    def temp_GetData(self):
        start = dt.datetime(2017, 1, 1)
        end = dt.datetime.now() - timedelta(days = 1)
        print(self.tickers)
        if len(self.tickers) == 0:
            messagebox.showinfo("Ticker List Underflow", "Retrieve latest S&P500 list using 'Get Tickers' button")
        else:
            if not os.path.exists('temp/stock_dfs'):
                os.makedirs('temp/stock_dfs')

            for ticker in self.tickers:
                if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                    df = web.DataReader(ticker, 'morningstar', start, end)
                    df.to_csv('temp/stock_dfs/{}.csv'.format(ticker))
                    time.sleep(1)

                else:
                    print('Data for {} already exists'.format(ticker))

    def visualize_data(self):


        df = pd.read_csv('sp500_closes.csv')

        df_corr = df.corr()
        print(df_corr.head())
        data = df_corr.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        heatmap = ax.pcolor(data, cmap = plt.cm.seismic)
        fig.colorbar(heatmap)
        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor = False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor = False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()


        column_labels = df_corr.columns
        row_labels = df_corr.index

        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(row_labels)
        plt.xticks(rotation = 90)
        heatmap.set_clim(-1, 1)
        plt.tight_layout()
        plt.show()

    def compile_data(self):
        main_df = pd.DataFrame()
        for count,ticker in enumerate(self.tickers):
            try:
                df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
                df.set_index('Date', inplace = True)

                df.rename(columns = {'Close':ticker}, inplace = True)
                df.drop(['Open','High','Low','Volume'], 1, inplace = True)

                if main_df.empty:
                    main_df = df
                else:
                    main_df = main_df.join(df)
            except:
                pass
            if count% 10 == 0:
                print(count)
        print(main_df.head)
        main_df.to_csv('sp500_closes.csv')

    def ticker_list(self):

        with open("SP500tickers.pickle", "rb") as f:
            ticks = pickle.load(f)
            root = tkinter.Tk()

            listBox = tkinter.Listbox(root)
            for count, t in enumerate(ticks):
                listBox.insert(count, t)

            listBox.pack()
            root.mainloop()


app = KOvi()
app.mainloop()
