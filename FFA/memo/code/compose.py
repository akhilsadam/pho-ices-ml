import importlib
import os
c = os.getcwd()
d = os.path.dirname(os.path.abspath(__file__))
e = f'{d}/../out'
p = f'{d}/../plots'

import code # TODO generalize path

tex = open(f'{d}/main.tex', 'r').read()

lines = tex.split('\n')
for i,line in enumerate(lines):
    if line.startswith('PYTHON'):
        sp = line.split(' ')
        name = sp[1]
        caption = code.__dict__[name]()
        lines[i] = r"""
\begin{figure}[htbp]
   \centering
    \includegraphics[width=12cm]{fig1.pdf}
    \caption{CAPTION}
    \label{fig:comparison}
\end{figure}
""".replace('fig1.pdf',f'{p}/{name}').replace('CAPTION',caption)


os.makedirs(e, exist_ok=True)
os.system(f'touch {e}/memo.tex')
with open(f'{e}/memo.tex', 'w') as f:
    f.write('\n'.join(lines))
# os.system(f'TEXINPUTS="./{d};{c}/texstyles//;%LocalAppData%/Programs/MiKTeX/tex//;$TEXINPUTS" bibtex {e}/memo.tex')
os.system(f'TEXINPUTS="./{d};./{p};{c}/texstyles//;%LocalAppData%/Programs/MiKTeX/tex//;$TEXINPUTS" pdflatex -interaction=nonstopmode -output-directory={e}/ {e}/memo.tex')