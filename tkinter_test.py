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
# matplotlib.use("TkAgg")
import time
import numpy as np
# import alpha_vantage
# from alpha_vantage.timeseries import TimeSeries
from matplotlib.backends.backend_tkagg import FigureCanvasAgg,NavigationToolbar2TkAgg
from matplotlib.figure import Figure

style.use('ggplot')
fontsets = ("Verdana", 12)

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
        btnStart = ttk.Button(self, text="Start", command=lambda: controller.show_frame(HomePage))
        btnStart.pack()


class HomePage(tkinter.Frame):

    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        self.tickers = []
        try:
            with open("SP500tickers.pickle", "rb") as f:
                self.tickers = pickle.load(f)

        except:
            self.getSP500()

        l1 = tkinter.Label(self, text="Home Page", font= fontsets)
        l1.pack(padx=10, pady=10)

        btnHome = ttk.Button(self, text="Return", command = lambda: controller.show_frame(StartPage))
        btnHome.pack()

        btnGetData = ttk.Button(self, text="Get Data", command=self.GetData)
        btnGetData.pack()

        btnGetTickers = ttk.Button(self, text="Get Tickers", command=self.getSP500)
        btnGetTickers.pack()

        btnCombineData = ttk.Button(self, text="Compile Data", command=self.compile_data)
        btnCombineData.pack()

        btnMorningstar = ttk.Button(self, text="Morningstar Data", command=self.temp_GetData)
        btnMorningstar.pack()

        btnCorTable = ttk.Button(self, text="Corelation Table", command=self.visualize_data )
        btnCorTable.pack()

        btnVisualiseTest = ttk.Button(self, text="Graph Test", command=self.live_graph)
        btnVisualiseTest.pack()
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
        end = dt.datetime.now() - timedelta(days=1)
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
        end = dt.datetime.now() - timedelta(days=1)
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

    def live_graph(self):
        df=pd.read_csv('sp500_closes.csv')
        df['A'].plot()
        plt.show()


    def visualize_data(self):


        df = pd.read_csv('sp500_closes.csv')

        df_corr=df.corr()
        print(df_corr.head())
        data=df_corr.values
        fig=plt.figure()
        ax= fig.add_subplot(111)
        heatmap=ax.pcolor(data, cmap=plt.cm.seismic)
        fig.colorbar(heatmap)
        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()


        column_labels =df_corr.columns
        row_labels =df_corr.index

        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(row_labels)
        plt.xticks(rotation=90)
        heatmap.set_clim(-1,1)
        plt.tight_layout()
        plt.show()


    def compile_data(self):
        main_df = pd.DataFrame()
        for count,ticker in enumerate(self.tickers):
            try:
                df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
                df.set_index('Date',inplace=True)

                df.rename(columns={'Close':ticker}, inplace=True)
                df.drop(['Open','High','Low','Volume'], 1, inplace=True)

                if main_df.empty:
                    main_df=df
                else:
                    main_df= main_df.join(df)
            except:
                pass
            if count% 10==0:
                print(count)
        print(main_df.head)
        main_df.to_csv('sp500_closes.csv')


class GraphPage(tkinter.Frame):
    def __init__(self, parent, controller):
        tkinter.Frame.__init__(self, parent)

        l1 = tkinter.Label(self, text="Graph", font=fontsets)
        l1.pack(padx=10, pady=10)
        btnHome = ttk.Button(self, text="Start", command=lambda: controller.show_frame(StartPage))
        btnHome.pack()

app = KOvi()
app.mainloop()
