# v1.2 수수료 기반 백테스팅 작업, AI 이용한 upper, lower 예측(분류모델 + gridcv로 최적 파라미터 설정), 코드 정리 
# v1.1 확대 축소가 아닌 Low, Upper에 닿을 경우 매수, 매도 결정, 5일 이동평균선 기준
# v1.0 볼린저 밴드 (확대, 축소)를 이용한 알고리즘 매수, 매도 결정

import pandas as pd
import numpy as np
import pyupbit
import datetime
import time
import os

# key numbers
access_key='RxxAtjdh5TCBv4Wc9ycWKILVB1sDhE3Z0eiJV4Xad8xF' # key 비식별화
secret_key='xavsrBGokR5fqeivyA4ctxFrJgMfjbCCJ1mx048RY5I5' # key 비식별화

upbit=pyupbit.Upbit(access_key,secret_key) # upbit 함수 이용하기

# 초기 데이터 추천
def start_settings(tickers):
    # 6시간 데이터 확보
    data = pyupbit.get_ohlcv(tickers, interval="minute1", count=60)  # 60분 추출
    i=1
    while(i<6): # 데이터 수 확장하기  (6시간 데이터 추출)
            date = data.index[0]
            data2 = pyupbit.get_ohlcv(tickers, interval="minute1", count=60, to=date)
            data = pd.concat([data,data2], axis=0)  
            data = data.sort_index()
            i+=1
            time.sleep(0.1)
    return data

# Data 확보
def settings(tickers):
    # 48시간 데이터 확보
    data = pyupbit.get_ohlcv(tickers, interval="minute1", count=60)  # 60분 추출
    i=1
    while(i<72): # 데이터 수 확장하기  (72시간 데이터 추출) # 3일치
            date = data.index[0]
            data2 = pyupbit.get_ohlcv(tickers, interval="minute1", count=60, to=date)
            data = pd.concat([data,data2], axis=0)  
            data = data.sort_index()
            i+=1
            time.sleep(0.1)
    return data

# 볼린저밴드 만들기(upper, middle, lower)
def BAND_data(df):
    
    #종가를 이용
    df['line_5']=df['close'].rolling(window=5).mean() # 5일 이동평균선
    df['line_10']=df['close'].rolling(window=10).mean() # 10일 이동평균선
    df['band_mid']=df['close'].rolling(window=20).mean() # 20일 이동평균선
    df['band_up']=df['band_mid']+(df['close'].rolling(window=20).std()*2)  # upper
    df['band_low']=df['band_mid']-(df['close'].rolling(window=20).std()*2)  # lower
    df=df.fillna(0) # 없는 데이터 채우기
    return df

#양봉, 음봉 캔들 생성
def minute_1(temp_df, i):
    blue_b_price=0
    blue_t_price=0
    blue=0
    red_b_price=0
    red_t_price=0
    red=0
    
    # 일봉 매수, 매도 점 위치 파악 (빨간색, 파란색인지 구별) -> 가격 변수 설정
    chai= float(np.round(temp_df.loc[temp_df.index[i],'open'],0) - np.round(temp_df.loc[temp_df.index[i],'close'],0))
    if chai >= 0.0: # 음봉
        blue_b_price=np.round(temp_df.loc[temp_df.index[i],'close'],0)
        blue_t_price=np.round(temp_df.loc[temp_df.index[i],'open'],0)
        blue=1
    else: #양봉
        red_b_price=np.round(temp_df.loc[temp_df.index[i],'open'],0)
        red_t_price=np.round(temp_df.loc[temp_df.index[i],'close'],0)   
        red=1
    
    return blue_b_price, blue_t_price, blue, red_b_price, red_t_price, red

# 볼린저 밴드를 이용한 매수 매도 누적 수익률 test 함수
def band_trainig(temp_df):
    temp_df['B/S']=0  # buy=1, sell=-1
    sign=0  # 매수 or 매도 타이밍
    cnt=0 # 매수 개수
    
    for i in range(1,len(temp_df)):
        mid_price=temp_df.loc[temp_df.index[i],'band_mid']
        upper=temp_df.loc[temp_df.index[i],'band_up']
        lower=temp_df.loc[temp_df.index[i],'band_low']
        line_5=temp_df.loc[temp_df.index[i],'line_5']
        blue=0
        red=0
        
        # 일봉 매수, 매도 점 위치 파악 (빨간색, 파란색인지 구별) -> 가격 변수 설정
        blue_b_price, blue_t_price, blue, red_b_price, red_t_price, red = minute_1(temp_df, i)
            
        if blue==1 and upper<blue_t_price and mid_price<blue_t_price:   #블루봉 사인
            sign=-1 # 매도 타이밍 발생
        elif blue==1 and lower>blue_b_price and mid_price>blue_b_price: 
            sign=1 # 매수 타이밍 발생(1)
        elif red==1 and upper<red_t_price and mid_price<red_t_price:    #레드봉 사인
            sign=-1 # 매도 타이밍 발생(-1)
        elif red==1 and lower>red_b_price and mid_price>red_b_price:
            sign=1 # 매수 타이밍 발생
        
        if sign==-1: # 시그널 지속하며 꺾는점 발생 (진정한 매도 타이밍)
            if blue==1 and blue_t_price<line_5 and mid_price<blue_b_price and cnt>0: # 고점형성 기준 블루봉일시 매도
                temp_df.loc[temp_df.index[i],'B/S']=-1   # 매도한다. 
                cnt=0
                sign=0

            elif red==1 and red_t_price<line_5 and mid_price<red_b_price and cnt>0: # 고점형성 기준 레드봉일시 매도
                temp_df.loc[temp_df.index[i],'B/S']=-1   # 매도한다. 
                cnt=0
                sign=0
                
        elif sign==1: # 시그널 지속하며 꺾는점 발생 (진정한 매수 타이밍)
            if red==1 and mid_price>red_b_price and red_b_price>line_5: # 저점형성 기준 레드봉일시 매수 (레드봉일시만 매수!! (핵심))
                temp_df.loc[temp_df.index[i],'B/S']=1   # 매수한다. 
                cnt+=1
                sign=0
    return temp_df

#누적수익률 계산 함수
def test_rate(temp_df):
    price=0
    cnt=0 # 처음 개수
    sell_rate=[] # 기간별 수익률
    cnt_sum=[]
    total_rate=1
    fee=0.0005  # upbit 수수료 고려
    for i in range(1,len(temp_df)):
        # 일봉 매수, 매도 점 위치 파악 (빨간색, 파란색인지 구별) -> 가격 변수 설정
        blue_b_price, blue_t_price, blue, red_b_price, red_t_price, red = minute_1(temp_df, i)
        
        if temp_df.loc[temp_df.index[i],'B/S']==1 and red==1:   # 매수는 시가로 매수한다. (red일시만 매수)
            price=(price+(red_b_price*(1.0+fee))) # red일 시 아랫 부분
            cnt+=1
        elif temp_df.loc[temp_df.index[i],'B/S']==-1 and cnt>0  and (blue==1 or red==1): # 매도
            if blue==1:  # 블루일시 윗 부분
                sell_p=(blue_t_price*cnt*(1.0-fee))
            elif red==1: # 레드일시 윗 부분
                sell_p=(red_t_price*cnt*(1.0-fee))
            rate=np.round(((sell_p-price)/price),4)
            sell_rate.append(rate)
            cnt_sum.append(cnt)
            price=0
            cnt=0
    # 총 누적수익률
    try:
        for rate in sell_rate:
            total_rate*=(1+rate)
        total_rate-=1
        total_rate=np.round(total_rate*100,5)
        if len(cnt_sum)!=0:
            cnt_max=max(cnt_sum)
        else:
            cnt_max=0
    except:
        cnt_max='error'
        total_rate='error'
    return total_rate, cnt_max

def not_trade(tickers, time2): # 미체결된 주문 취소
    ret=upbit.get_order(tickers, state='wait')
    i=0
    while len(ret)!=0:
        ret=upbit.get_order(tickers)
        uuid=ret[i]['uuid']
        ret2=upbit.cancel_order(uuid)
        print('{0} 미체결된 주문 {2} 중 {1}원이 취소되었습니다.'.format(time2,ret2['price'],tickers[4:]))
        i+=1

def buy_trade(tickers, time2, cnt, buy_cnt): #매수 함수
    try:
        ret=upbit.get_balances()
        check=0
        for name in ret:
            if name['currency']==tickers[4:]:  # 잔고에 있는 매수가 확인
                if float(name['avg_buy_price'])>price:
                    check=0
                else:
                    check=1
        price=pyupbit.get_current_price(tickers)
        if check==1:
            print('잔고에 있는 평균가 {0}원이 주문가격 {1}원보다 낮아 매수 주문이 취소되었습니다.'.format(name['avg_buy_price'],price))
        else: 
            ret = upbit.buy_market_order(tickers, price*buy_cnt) # 시장가 매수를 가격으로 한다.
            cnt+=1
            print('{1} 으로 {2}는 {0}원으로 현재 {4}{3}를 매수 주문 완료했습니다.'.format(price,time2,tickers,tickers[4:],buy_cnt)) 
        return cnt, price
    except:
        print(ret)
        return cnt, price

def sell_trade(tickers, time2, cnt, buy_cnt): #매도 함수   
    try:
        ret=upbit.get_balances()
        for name in ret:
            if name['currency']==tickers[4:]:  # 잔고에 있는 매수평균가 서칭
                avg_buy_price=float(name['avg_buy_price'])
                ret = upbit.sell_market_order(tickers, buy_cnt*cnt)
                price=pyupbit.get_current_price(tickers)
                while True:
                    if len(upbit.get_order(tickers, state='wait'))==0:
                        print('{1} 기준으로 {2}는 {0}원으로 주문({3}) 만큼 매도 완료했습니다.'.format(price,time2,tickers[4:],buy_cnt))
                        avg_price=np.round(((price-avg_buy_price)/avg_buy_price*100),2)
                        print('{0} 수익률은 {1}% 입니다.'.format(time2,avg_price))
                        cnt=0
                        break
        return cnt, price
    except:
        print(ret)
        return cnt, price

# 매수 or 매도 프로그램
def trade(tickers, df, buy_cnt, sign, cnt): # sign=매수 or 매도 타이밍 # band_trainig 에서 결정된 signal 값
    
    #최신데이터 band 형성
    df=BAND_data(df)
    line_5=df.loc[df.index[-1],'line_5']
    line_10=df.loc[df.index[-1],'line_5']
    upper=df.loc[df.index[-1],'band_up']
    lower=df.loc[df.index[-1],'band_low']
    df.loc[df.index[-1],'B/S']=0
    blue=0
    red=0
    
    # 일봉 매수, 매도 점 위치 파악 (빨간색, 파란색인지 구별) -> 가격 변수 설정
    blue_b_price, blue_t_price, blue, red_b_price, red_t_price, red = minute_1(df, -1)
    
    #변수 설정
    price=pyupbit.get_current_price(tickers)  #현재 가격
    time2=df.index[-1]  #시간
    avg_buy_price=0.0
    avg_price=0.0
    
    #band에 따른 매수 or 매도 결정
    mid_price=df.loc[df.index[-1],'band_mid']
    
    # signal 발생
    if blue==1 and upper<blue_t_price and mid_price<blue_t_price:   #블루봉 사인
        sign=-1 # 매도 타이밍 발생
    elif blue==1 and lower>blue_b_price and mid_price>blue_b_price: 
        sign=1 # 매수 타이밍 발생(1)
    elif red==1 and upper<red_t_price and mid_price<red_t_price:    #레드봉 사인
        sign=-1 # 매도 타이밍 발생(-1)
    elif red==1 and lower>red_b_price and mid_price>red_b_price:
        sign=1 # 매수 타이밍 발생
    
    if sign==-1 and cnt>0: # 시그널 지속하며 꺾는점 발생 (진정한 매도 타이밍)
        if blue==1 and blue_t_price<line_5 and mid_price<blue_b_price: # 고점형성 기준 블루봉일시 매도
            df.loc[df.index[-1],'B/S']=-1   # 매도한다. 
            cnt, price = sell_trade(tickers, time2, cnt, buy_cnt) # 시장가 매도
            sign=0

        elif red==1 and red_t_price<line_5 and mid_price<red_b_price: # 고점형성 기준 레드봉일시 매도
            df.loc[df.index[-1],'B/S']=-1   # 매도한다. 
            cnt, price = sell_trade(tickers, time2, cnt, buy_cnt) # 시장가 매도
            sign=0
            
        else:
            print('{1} 기준으로 {2}는 {0}원으로 현재 보류중입니다.(signal 보류 중)'.format(price,time2,tickers))

    elif sign==1: # 시그널 지속하며 꺾는점 발생 (진정한 매수 타이밍)
        if red==1 and mid_price>red_b_price and red_b_price>line_5: # 저점형성 기준 레드봉일시 매수 (레드봉일시만 매수!! (핵심))
            df.loc[df.index[-1],'B/S']=1   # 매수한다. 
            cnt, price = buy_trade(tickers, time2, cnt, buy_cnt)  # 시장가 매수
            sign=0
            
        else:
            print('{1} 기준으로 {2}는 {0}원으로 현재 보류중입니다.(signal 보류 중)'.format(price,time2,tickers))
    
    else:
        print('{1} 기준으로 {2}는 {0}원으로 현재 보류중입니다.'.format(price,time2,tickers))
        
    return sign, cnt, df
        
# 반복 함수 설정
def repeat_module(tickers, band_df, start_rate, buy_cnt):
    sign=0   # band 발생 표시 여부
    cnt=0    # 중복 매수 여부
    while True:
        try:
            time2=datetime.datetime.now()
            if time2.second==1:
                not_trade(tickers, time2)  # 미체결된 주문 취소
                time.sleep(1)
                temp = pyupbit.get_ohlcv(tickers, interval="minute1", count=1)
                band_df=pd.concat([band_df,temp], axis=0)
                band_df.drop([band_df.index[0]], inplace=True)
                sign, cnt, band_df=trade(tickers, band_df, buy_cnt, sign, cnt)
                rate, cnt_max = test_rate(band_df)
                #print("개발자 전용: test= {0}%".format(rate))
                if rate<0:  # 음수일 경우만 수익률 조정
                    print("-----------------Program Pause------------------------")
                    print("   예상수익률 허용범위 벗어나 시스템 재가동 필요      ")
                    print("                                                      ")
                    print("   현재 매수되어 있는 수량: {0}개".format(cnt))
                    print("-----------------Program Pause------------------------")
                    tickers, band_df, start_rate, buy_cnt=slot_setting(0,tickers)

        except KeyboardInterrupt:   # 무한반복 종료
            print("-------------------------Program Pause-------------------------")
            print("       중복 키 입력으로 시스템 중지 및 초기화면 이동           ")
            print("                                                               ")
            print("       현재 매수되어 있는 수량: {0}개".format(cnt))
            print("-------------------------Program Pause-------------------------")
            os.system("pause")
            os.system("cls")
            tickers, band_df, start_rate, buy_cnt=slot_setting(0,tickers)
            
# 초기 실행 시 6시간 수익률이 높은 것 top10 종목 선정
def choose_select(tickers_list):
    dic_rate={}
    for coin in tickers_list:
        data=start_settings(coin)
        data=BAND_data(data)
        data=band_trainig(data)
        temp_rate, cnt_max=test_rate(data)
        dic_rate[coin]=np.round(temp_rate,2)
    df_rate=pd.DataFrame(list(dic_rate.items()), columns=['coin','rate'])
    df_rate=df_rate.sort_values(by='rate', ascending=False).reset_index(drop=True)
    df_rate=df_rate[:10]
    return df_rate

# 초기 실행함수
def start():
    tickers_list = pyupbit.get_tickers(fiat="KRW")
    print("")
    print("-----------Upbit Coin Auto control Program------------------------")
    print("|  Version. 1.2                                                  |")
    print("|                                                                |")
    print("|  코인명 작성법: 'KRW-<코인명>'                                 |")
    print("|                                  ex) KRW-EOS  (이오스)         |")
    print("|  참고) 프로그램은 매 1분마다 스스로 작동합니다.                |")
    print("|  참고2) 실행도중 초기화면 이동을 원할 경우 Ctrl+C 입력         |")
    print("|  참고3) 예상수익률은 누적수익률로 계산되었습니다.              |")
    print("|  참고4) 수익률은 과거 72시간 Data를 기반으로 작성되었습니다.   |")
    print("----------------------------------------------made.by Lutto-------\n\n\n")
    money=np.round(upbit.get_balance("KRW"),0)
    choice=input("6시간 수익률 중 Top 10 종목 선정이 필요하시나요? (3분이상 소요)   [y / n]  ")
    if choice=='y':
        df=choose_select(tickers_list)
        print("   종목명\t\t\t6시간 수익률")
        for i in range(len(df)):
            print("  {0}\t\t\t  {1}%".format(df.loc[i,'coin'],df.loc[i,'rate']))
        print(" ")
        
    print(f'현재 잔고 : {money} won')
    Coin=input("Select User Coin: ")
    if Coin in tickers_list:
        print('\n',Coin,'is checked')
        current=datetime.datetime.now()
        print("현재 접속 시간:{0}".format(current.strftime("%Y-%m-%d %H:%M")))
        time.sleep(2)
        return Coin
    else:
        print("잘못된 코인명을 입력했습니다. 코인명을 아래에서 확인 후 원하는 것을 골라 입력해주세요.")
        display(tickers_list)
        os.system("pause")
        os.system("cls")
        Coin=start()
        return Coin

#초기화 함수
def slot_setting(dummy,tickers):
    if dummy==0:   # 예상 수익률의 극심한 변동으로 초기화 세팅 진행할 경우
        tickers =start()
        present_p=pyupbit.get_current_price(tickers)
        limit_order=np.round(5000/present_p,4)
        print("{0} 현재 가격 : {1}원, 최소 주문 요구 코인 수량: {2}개 이상".format(tickers, present_p, limit_order))
        buy_cnt=float(input("한번 매수시 구매할 코인 수량: "))
        
    print("==================== 초기화 세팅 진행 ====================")
    print("  0. 데이터 생성:   ",end='')
    data = settings(tickers)
    print(" CLEAR")
    print("  1. 72시간 사용 시 예상 수익률:   ",end='')
    band_df=BAND_data(data)
    band_df=band_trainig(band_df)
    rate, cnt_max=test_rate(band_df)   #72시간 기준 signal 사용
    print(" {0}%".format(rate))
    print("  2. 6시간 사용 시 예상 수익률:   ",end='')
    #band_df.to_csv('test.csv')
    temp_df = band_df[(len(band_df)-360):]  # 6시간 예상 수익률
    rate2, cnt_max2=test_rate(temp_df)
    print(" {0}%".format(rate2))
    print("  3. 6시간 사용 시 중복 매수 최대 횟수:  {0}번".format(cnt_max2))
    print("==========================================================")
    time2=datetime.datetime.now()
    print("{0}초 후 프로그램 시작".format(60-time2.second))
    return tickers, band_df, rate, buy_cnt
    
if __name__ == "__main__":
    dummy=0 # 0일시 완전 새로 시작
    try:
        tickers, band_df, start_rate, buy_cnt=slot_setting(dummy,'')
        repeat_module(tickers, band_df, start_rate, buy_cnt)
    except:
        print("Program Error")
        os.system("pause")
