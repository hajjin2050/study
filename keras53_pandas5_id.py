import pandas as pd

df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]],
                  columns = list('abcd'), index=('가','나','다')) # 판다스는 열이 우선
print(df)
#    a  b  c   d
# 가  1  2  3   4
# 나  4  5  6   7
# 다  7  8  9  10
df2 = df
df2['a'] = 100
print(df2)

print(df)
#      a  b  c   d
# 가  100  2  3   4
# 나  100  5  6   7
# 다  100  8  9  10
#      a  b  c   d
# 가  100  2  3   4
# 나  100  5  6   7
# 다  100  8  9  10

print(id(df), id(df2)) #2411005290576 2411005290576 => 동일메모리를 사용한다.

df3 = df.copy() #새로운 인스턴스 생성
df2['b'] = 333

print("=================")
print(df)
print(df2)
print(df3)

df = df + 99 #=> 사칙연산, .copy 의 경우 새로운 df 저장
print(df)
print(df2)