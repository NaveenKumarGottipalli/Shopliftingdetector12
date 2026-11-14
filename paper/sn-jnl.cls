%% Clean Springer Nature Journal Class (sn-jnl.cls)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2026/11/14 v1.3 Springer Nature Journal Article Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[10pt,a4paper]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}
\RequirePackage[margin=2.4cm,top=2.5cm,bottom=2.5cm]{geometry}
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{url}
\RequirePackage{xcolor}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{fancyhdr}
\RequirePackage{caption}

\definecolor{snblue}{HTML}{004B87}
\RequirePackage[colorlinks=true,linkcolor=snblue,citecolor=snblue,urlcolor=snblue]{hyperref}

\captionsetup{font=small,labelfont=bf,labelsep=period}
\let\botrule\bottomrule

\newcommand{\fnm}[1]{#1}
\newcommand{\sur}[1]{#1}
\newcommand{\spfx}[1]{#1}
\newcommand{\sfx}[1]{#1}
\newcommand{\dgr}[1]{#1}

\newcommand{\@authorlist}{}
\newcommand{\@emaillist}{}
\newcommand{\@affillist}{}

\renewcommand{\author}{\@ifstar\@authorstar\@authornostar}
\newcommand{\@authorstar}[2][]{\g@addto@macro\@authorlist{\textbf{#2}\textsuperscript{#1,*}}}
\newcommand{\@authornostar}[2][]{\g@addto@macro\@authorlist{\textbf{#2}\textsuperscript{#1}}}

\newcommand{\email}[1]{\g@addto@macro\@emaillist{\texttt{#1}}}

\newcommand{\affil}{\@ifstar\@affilstar\@affilnostar}
\newcommand{\@affilstar}[2][]{\g@addto@macro\@affillist{\par\textsuperscript{#1}#2}}
\newcommand{\@affilnostar}[2][]{\g@addto@macro\@affillist{\par\textsuperscript{#1}#2}}

\newcommand{\orgdiv}[1]{#1}
\newcommand{\orgname}[1]{#1}
\newcommand{\orgaddress}[1]{#1}
\newcommand{\street}[1]{#1}
\newcommand{\city}[1]{#1}
\newcommand{\postcode}[1]{#1}
\newcommand{\state}[1]{#1}
\newcommand{\country}[1]{#1}

\newcommand{\@abstracttext}{}
\renewcommand{\abstract}[1]{\renewcommand{\@abstracttext}{#1}}

\newcommand{\@keywordstext}{}
\newcommand{\keywords}[1]{\renewcommand{\@keywordstext}{#1}}

\let\@oldtitle\title
\renewcommand{\title}[2][]{\gdef\@shorttitle{#1}\gdef\@maintitle{#2}}

\renewcommand{\maketitle}{%
  \vspace*{-0.8cm}%
  \begin{flushleft}%
    {\color{snblue}\rule{\textwidth}{1.5pt}}\par\vspace{0.35cm}%
    {\LARGE\bfseries\raggedright \@maintitle \par}%
    \vspace{0.35cm}%
    {\normalsize \@authorlist \par}%
    \vspace{0.15cm}%
    {\small\itshape \@affillist \par}%
    \vspace{0.15cm}%
    {\small \textbf{*Corresponding author:} \@emaillist \par}%
    \vspace{0.35cm}%
    {\color{black!25}\rule{\textwidth}{0.5pt}}\par\vspace{0.25cm}%
    {\small\textbf{Abstract}\quad \@abstracttext \par}%
    \vspace{0.2cm}%
    {\small\textbf{Keywords:}\quad \@keywordstext \par}%
    \vspace{0.25cm}%
    {\color{snblue}\rule{\textwidth}{1.0pt}}%
  \end{flushleft}%
  \vspace{0.3cm}%
}

\newcommand{\backmatter}{\par\vspace{0.4cm}}
\newcommand{\bmhead}[1]{\par\vspace{0.25cm}\noindent\textbf{#1.}\quad}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape \@shorttitle}
\fancyhead[R]{\small Gottipalli}
\fancyfoot[C]{\small \thepage}
\renewcommand{\headrulewidth}{0.4pt}

\bibliographystyle{unsrtnat}
