---
title: 编码AST树为数学符号计算数据结构
authors: Complex System Explorer
year: 2026-01-10
tags:
  - 类型/笔记
  - 日期/2026-01-10
alias:
  - 编码AST树为数学符号计算数据结构
---
# 编码AST树为数学符号计算数据结构


### SymPy解析

对于`"\frac{2x+1}{y z_{\beta}} + 3\cdot(x-1) + \alpha"`，经由SymPy解析之后，得到AST如下：

```text
AST 表达式: alpha + (3*(x - 1) + (2*x + 1)/((y*z_{beta})))
AST 类型: <class 'sympy.core.add.Add'>
类型: <class 'sympy.core.add.Add'>
参数: (3*(x - 1) + (2*x + 1)/((y*z_{beta})), alpha)
类型: <class 'sympy.core.add.Add'>, 内容: alpha + (3*(x - 1) + (2*x + 1)/((y*z_{beta})))
  类型: <class 'sympy.core.add.Add'>, 内容: 3*(x - 1) + (2*x + 1)/((y*z_{beta}))
    类型: <class 'sympy.core.mul.Mul'>, 内容: (2*x + 1)/((y*z_{beta}))
      类型: <class 'sympy.core.add.Add'>, 内容: 2*x + 1
        类型: <class 'sympy.core.mul.Mul'>, 内容: 2*x
          类型: <class 'sympy.core.numbers.Integer'>, 内容: 2
          类型: <class 'sympy.core.symbol.Symbol'>, 内容: x
        类型: <class 'sympy.core.numbers.One'>, 内容: 1
      类型: <class 'sympy.core.power.Pow'>, 内容: 1/(y*z_{beta})
        类型: <class 'sympy.core.mul.Mul'>, 内容: y*z_{beta}
          类型: <class 'sympy.core.symbol.Symbol'>, 内容: y
          类型: <class 'sympy.core.symbol.Symbol'>, 内容: z_{beta}
        类型: <class 'sympy.core.numbers.NegativeOne'>, 内容: -1
    类型: <class 'sympy.core.mul.Mul'>, 内容: 3*(x - 1)
      类型: <class 'sympy.core.numbers.Integer'>, 内容: 3
      类型: <class 'sympy.core.add.Add'>, 内容: x - 1
        类型: <class 'sympy.core.symbol.Symbol'>, 内容: x
        类型: <class 'sympy.core.numbers.NegativeOne'>, 内容: -1
  类型: <class 'sympy.core.symbol.Symbol'>, 内容: alpha
```




