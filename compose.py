import importlib
import os,sys,numpy as np
path = sys.argv[1] if len(sys.argv)>1 else 'FFA/memo'
c = os.getcwd()
d = f'{path}/code/'
e = f'{d}/../out'
p = f'{path}/plots'

code = importlib.import_module(f'{d}code'.replace('../','..').replace('/','.'),package='PHO_ICES_ML') # TODO generalize path

tex = open(f'{d}/main.tex', 'r').read()

tex = r"""\documentclass[12pt]{iopart}
\usepackage{graphicx}
%\usepackage{witharrows}
\usepackage{extarrows}
\usepackage{mathtools}
\usepackage{amssymb}
\usepackage{units}
\usepackage{tabto}
\usepackage{subfig}
\usepackage{ntheorem}
\theoremstyle{break}
\newtheorem{prop}{Definition}[section]
\counterwithin{equation}{section}
\DeclareMathOperator{\E}{\mathbb{E}}
\DeclareMathOperator{\sign}{sign}
\newcommand*{\QEDA}{\null\nobreak\hfill\ensuremath{\diamond}}%
\newcommand*{\QEDB}{\null\nobreak\hfill\ensuremath{\blacksquare}}%
\newcommand*{\sfig}[2]{\begin{figure}[htpb]\centering\subfloat[#2]{\includegraphics[width=1.0\linewidth]{#1}}\end{figure}}
\bibliographystyle{iopart-num}""" + '\n' + tex

tex = tex.replace('|-',r'\begin{description}\item').replace('-|',r'\end{description}') \
    .replace(r'\[',r'\begin{alignat}{2}').replace(r'\]',r'\end{alignat}')
    # .replace(r'\[',r'\begin{equation}').replace(r'\]',r'\end{equation}')

lines = tex.split('\n')
for i,line in enumerate(lines):
    fe = line[:2] if len(line)>1 else ' '*2
    if line.startswith('PYTHON'):
        sp = line.split(' ')
        name = sp[1:]
        # size = 12 / n
        outs = [code.__dict__[ni](f'{p}/{ni}.pdf') for ni in name]
        caption = [out[0] for out in outs]
        size = np.array([out[1] for out in outs])
        items = [(r"""
\subfloat[CAPTION]{
\includegraphics[width="""+str(sz)+r"""\linewidth]{fig1.pdf}
\label{fig:comparison}
}
""").replace('fig1.pdf',f'{p}/{ni}').replace('CAPTION',ci).replace('comparison', ni) for sz,ci,ni in zip(size,caption,name)]
        lines[i] = r"""
\begin{figure}[htpb]
\centering
"""+r'\qquad'.join(items)+r"""
\end{figure}
"""
    elif fe[0]==' ':
        lines[i] = '\t' + lines[i]
    elif fe[0]=='+':
        lines[i] = r'\begin{itemize}\item' + lines[i][1:] + r'\end{itemize}'
    # elif fe=='!!':
    #     lines[i] = r'\begin{empheq}' + lines[i][2:] + r'\end{empheq}'
    # elif fe=='|-':
    #     lines[i] = r'\par' + r'\hangindent=1in' + lines[i][2:]
    # elif fe=='-|':
    #     lines[i] = r'\hangindent' + lines[i][2:] + r'\par'
os.makedirs(e, exist_ok=True)
os.system(f'touch {e}/memo.tex')
with open(f'{e}/memo.tex', 'w') as f:
    f.write('\n'.join(lines))
# os.system(f'TEXINPUTS="./{d};{c}/texstyles//;%LocalAppData%/Programs/MiKTeX/tex//;$TEXINPUTS" bibtex {e}/memo.tex')
os.system(f'TEXINPUTS="./{d};./{p};{c}/texstyles//;%LocalAppData%/Programs/MiKTeX/tex//;$TEXINPUTS" xelatex -interaction=nonstopmode -output-directory={e}/ {e}/memo.tex')