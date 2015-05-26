%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  FILE: tudelft.cls                                                       %%
%%                                                                          %%
%%  This class can be used for writing your thesis in LaTeX.                %%
%%  It is based on the memoir class, which it uses extensively.             %%
%%  Both the class and the sample thesis are fully documented, feel free to %%
%%  look around and change things to your liking.                           %%
%%                                                                          %%
%%  Created by: Stavros Christoforou, s.christoforou@tudelft.nl             %%
%%                                                                          %%
%%  First public version : 27 November 2007                                 %%
%%  14/01/2008: Changed some Hyperref options to remove annoying warnings   %%
%%  17/01/2008: Added TUDelft chapter style, dedication bookmark            %%
%%  22/01/2008: Fixed error when frontmatter is commented out               %%
%%  29/01/2008: Fixed garbled page numbering in Acrobat when pdfTeX is used %%
%%  20/05/2008: Added chapterbib package                                    %%
%%  29/09/2008: Fixed several badboxes in frontmatter                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tudelft}[2007/11/27 TU Delft Thesis Class for LaTeX2e]

%% ========================== Class declarations ========================== %%

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions
\LoadClass{memoir}

%% =========================== Packages used ============================== %%

\RequirePackage{ucs}
\RequirePackage[utf8x]{inputenc}
\RequirePackage[dutch,british]{babel}                % To use English hyphenation.
\usepackage{chapterbib}                        % Per chapter bibliography
\usepackage[numbers,super,comma,sectionbib]{natbib}
\usepackage{amsmath,amstext}
\usepackage{multicol,multirow}
\usepackage{hypernat}
\usepackage{microtype}
\usepackage{type1cm}                    % Scale Computer Modern fonts properly
\usepackage{tocloft}
\RequirePackage[pdftex]{graphicx}
\RequirePackage[pdftex,unicode,colorlinks=false,pdffitwindow=true,
                pdfpagemode=UseNone,pdfpagelabels,
                plainpages=false,breaklinks=true] {hyperref}
\fixpdflayout
\usepackage{epsfig}
\usepackage{memhfixc}

%% ============================ Page geometry ============================= %%

\setstocksize          {240mm}{170mm}      % Uncomment this, comment next two lines and remove 'showtrims' class option to see the final size of your thesis
%\setstocksize          {297mm}{210mm}      % A4 page size
%\settrims              {28.5mm}{20mm}
\settrimmedsize        {240mm}{170mm}{*}   % Thesis page size
\settypeblocksize      {210mm}{130mm}{*}   % Text size
\setlrmarginsandblock  {20mm}{*}{*}
\setulmarginsandblock  {28mm}{32mm}{*}
\setheadfoot           {5mm}{11mm}
\setheaderspaces       {*}{5.5mm}{*}
\setlength{\parindent} {2em}
\setlength{\parskip}   {0.5\baselineskip}
\checkandfixthelayout

% numbering of chapter, section, table, figure, and equation:
\renewcommand{\thechapter}      {\arabic{chapter}}
\renewcommand{\thesection}      {\thechapter.\arabic{section}}
\renewcommand{\thesubsection}   {\thesection.\arabic{subsection}}
\renewcommand{\thetable}        {\thechapter.\arabic{table}}
\renewcommand{\thefigure}       {\thechapter.\arabic{figure}}
\renewcommand{\theequation}     {\thechapter.\arabic{equation}}

% redefine some standard LaTeX words
\def\captionsenglish{
        \def\chaptername    {Chapter}
        \def\contentsname   {Contents}
        \def\appendixname   {Appendix}
        \def\bibname        {Bibliography}
}

%% =========================== General options ============================ %%

%\sloppy                                % suppress hyphenation
%\trimFrame                             % To use full lines for trimming
\trimXmarks                             % To use corners for trimming
\leavespergathering{8}					% If specific # of pages required by publisher

\maxsecnumdepth{subsection}             % Section numbering depth
\maxtocdepth{section}                   % Table of contents depth

\newcommand {\halfendline}  {\\[5pt]}   % Endlines
\newcommand {\fullendline}  {\\[11pt]}

\newcommand {\ptext}[1]     {\small #1} % text in figures slightly smaller
\def\pshlabel#1{\small #1}
\def\psvlabel#1{\small #1}

\newcommand{\tablesettings}{            % settings for a table
\renewcommand{\arraystretch}{1.3}
\renewcommand{\baselinestretch}{1.5}
\setlength\doublerulesep{0pt}
\doublerulesepcolor{black}\center
}

%% Ensure no bad boxes if page count reaches three figures
\renewcommand{\@pnumwidth}{2em}
\renewcommand{\@tocrmarg}{3em}


%% ============================= Page Style =============================== %%

\makeatletter
\setlength\midchapskip{10pt}

\makechapterstyle{TUDelft}{
\renewcommand\chapternamenum{}
\renewcommand\printchaptername{}
\renewcommand\chapnamefont{\Large\scshape\centering}
\renewcommand\printchapternum{%
\chapnamefont\null
\@chapapp\space\thechapter}
\renewcommand\printchapternonum{%
\par\vskip\midchapskip
\hrule\vskip\midchapskip}
\renewcommand\chaptitlefont{\Huge\scshape\centering}
\renewcommand\afterchapternum{%
\par\nobreak\vskip\midchapskip\hrule\vskip\midchapskip}
\renewcommand\afterchaptertitle{%
\par\vskip\midchapskip\hrule\nobreak\vskip\afterchapskip}
\setsecheadstyle{\Large\bfseries\raggedright}
\setsubsecheadstyle{\large\bfseries\raggedright}
}

\makepagestyle{TUDelft}
\makerunningwidth{TUDelft}{\textwidth}
\makeheadposition{TUDelft}{}{}{}{}
\makeheadrule{TUDelft}{\textwidth}{\normalrulethickness}
\makepsmarks{TUDelft}{\@ruledmarks}
\makeevenhead{TUDelft}{\normalfont\slshape\leftmark}{}{}
\makeoddhead{TUDelft}{}{}{\normalfont\slshape\rightmark}
\makeevenfoot{TUDelft}{\slshape\thepage}{}{}
\makeoddfoot{TUDelft}{}{}{\slshape\thepage}

\copypagestyle{TUDedication}{empty}
\makeevenfoot{TUDedication}{}{}{}
\makeoddfoot{TUDedication}{}{}{\slshape\thepage}

\copypagestyle{TUDchapter}{TUDelft}
\makeheadrule{TUDchapter}{\textwidth}{0em}
\makeoddfoot{TUDchapter}{}{}{\slshape\thepage}



%% ============================== Formatting ============================== %%

\let\thesis@frontmatter=\frontmatter
\def\frontmatter{
  \thesis@frontmatter
  \setlength{\parindent}{0em}             % Set indentation in frontmatter
  \setlength{\parskip}{0\baselineskip}
  \pagestyle{empty}
  \aliaspagestyle{chapter}{TUDelft}
  \aliaspagestyle{cleared}{TUDelft}
}


\let\thesis@mainmatter=\mainmatter
\def\mainmatter{
  \thesis@mainmatter
  \setlength{\parindent}{1em}             % Set indentation in main
  \setlength{\parskip}{0.5\baselineskip}
  \chapterstyle{TUDelft}
  \pagestyle{TUDelft}                   % Change to the pagestyle
  \aliaspagestyle{chapter}{TUDchapter}
  \aliaspagestyle{cleared}{TUDelft}
}

\let\thesis@backmatter=\backmatter
\def\backmatter{
  \thesis@backmatter
  \setlength{\parindent}{1em}             % Set indentation in backmatter
  \setlength{\parskip}{0.5\baselineskip}
  \chapterstyle{reparticle}
  \aliaspagestyle{cleared}{TUDelft}
}

%% ================================ Macros ================================ %%

% These macros are used to declare arguments needed for the
% construction of the preliminary pages

\newcommand{\firstname}[1]{\gdef\@firstname{#1}}
\newcommand{\lastname}[1]{\gdef\@lastname{#1}}
\newcommand{\authortitle}[1]{\gdef\@authortitle{#1}}
\newcommand{\birthplace}[1]{\gdef\@birthplace{#1}}
\newcommand{\titletop}[1]{\gdef\@titletop{\mbox{#1}}}
\newcommand{\titlemiddle}[1]{\gdef\@titlemiddle{\mbox{#1}}}
\newcommand{\titlebottom}[1]{\gdef\@titlebottom{\mbox{#1}}}
\newcommand{\defensedate}[1]{\gdef\@defensedate{#1}}
\newcommand{\defensetime}[1]{\gdef\@defensetime{#1}}
\newcommand{\firstpromotor}[1]{\gdef\@firstpromotor{#1}}
\newcommand{\secondpromotor}[1]{\gdef\@secondpromotor{#1}}
\newcommand{\rector}[1]{\gdef\@rector{#1}}
\newcommand{\@member}[1]{\raisebox{0pt}[0pt][0pt]{#1}}%,}
\newcommand{\@affiliation}[1]{\raisebox{0pt}[0pt][0pt]{#1}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\newcommand{\dedicationtop}[1]{\gdef\@dedicationtop{#1}}
\newcommand{\dedicationmiddle}[1]{\gdef\@dedicationmiddle{#1}}
\newcommand{\dedicationbottom}[1]{\gdef\@dedicationbottom{#1}}
\newcommand{\support}[1]{\gdef\@support{#1}}
\newcommand{\work}[1]{\gdef\@work{#1}}
\newcommand{\isbnno}[1]{\gdef\@isbnno{#1}}

\newcounter{@minmembers} \setcounter{@minmembers}{6}
\newcounter{@minaffiliations} \setcounter{@minaffiliations}{6}
\newcounter{@membersprinted} \setcounter{@membersprinted}{0}
\newcounter{@affiliationsprinted} \setcounter{@affiliationsprinted}{0}

%% Loops for committee

\newcounter{@members}
\newcommand{\member}[1]{
  \addtocounter{@members}{1}
  \ifcase\value{@members}
    \typeout{Thesis format internal error. Too few committee members.} \or
    \newcommand{\@membera}{\@member{#1}} \or
    \newcommand{\@memberb}{\@member{#1}} \or
    \newcommand{\@memberc}{\@member{#1}} \or
    \newcommand{\@memberd}{\@member{#1}} \or
    \newcommand{\@membere}{\@member{#1}} \or
    \newcommand{\@memberf}{\@member{#1}} \or
    \newcommand{\@memberg}{\@member{#1}} \or
    \newcommand{\@memberh}{\@member{#1}} \or
    \newcommand{\@memberi}{\@member{#1}} 
  \else
    \typeout{Error using thesis format.  Too many committee members.}
  \fi
}

\newcounter{@affiliations}
\newcommand{\affiliation}[1]{
  \addtocounter{@affiliations}{1}
  \ifcase\value{@affiliations}
    \typeout{Thesis format internal error. Too few committee affiliations.} \or
  \newcommand{\@affiliationa}{\@affiliation{#1}} \or
    \newcommand{\@affiliationb}{\@affiliation{#1}} \or
    \newcommand{\@affiliationc}{\@affiliation{#1}} \or
    \newcommand{\@affiliationd}{\@affiliation{#1}} \or
    \newcommand{\@affiliatione}{\@affiliation{#1}} \or
    \newcommand{\@affiliationf}{\@affiliation{#1}} \or
    \newcommand{\@affiliationg}{\@affiliation{#1}} \or
    \newcommand{\@affiliationh}{\@affiliation{#1}} \or
    \newcommand{\@affiliationi}{\@affiliation{#1}}
  \else
    \typeout{Error using thesis format.  Too many committee affiliations.}
  \fi
}

%% ============================ Title Page ================================ %%

\newcommand{\titlepage}{
  \thispagestyle{empty}

  \begin{centering}
  \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Main Title}{bm:MainTitle}
  \vspace*{0.375in}
  {\huge\textbf{
  \@titletop\\[0.25\baselineskip]
    \@titlemiddle\\[0.25\baselineskip]
    \@titlebottom\\[0.25\baselineskip]
  }} \vspace{35.0pt}
  \end{centering}
  \clearpage
  \thispagestyle{empty}
  \vspace*{1in}
  \clearpage

\begin{centering}
  \providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Title Page}{bm:Title}
  \vspace*{0.375in}
  {\huge\textbf{
  \@titletop\\[0.25\baselineskip]
    \@titlemiddle\\[0.25\baselineskip]
    \@titlebottom\\[0.25\baselineskip]
  }} \vspace{35.0pt}
  \textbf{Proefschrift} \\ \vspace{25.pt}
  ter verkrijging van de graad van doctor \halfendline
  aan de Technische Universtiteit Delft, \halfendline
  op gezag van de Rector Magnificus \@rector , \halfendline
  voorzitter van het College voor Promoties, \halfendline
  in het openbaar te verdedigen op \@defensedate\hspace{0pt} om \@defensetime\hspace{0pt} uur \\
  \vspace{25.0pt} door \\ \vspace{25.0pt}
  {\Large \textbf{\@firstname\hspace{0pt} \MakeUppercase{\@lastname}} }\\
  \vspace{25.0pt}
  \@authortitle \halfendline
  geboren te \@birthplace  \\
\end{centering}
\vspace{1cm}
\clearpage
}

%% ========================== Copyright page ============================== %%

\newcommand{\copyrightpage}{
\thispagestyle{empty}

\textsl{Dit proefschrift is goedgekeurd door de promotor:}\\[0.2\baselineskip]
\@firstpromotor\\[\baselineskip]

%\textsl{Copromotor:}\\[0.2\baselineskip]
%\@secondpromotor\\[\baselineskip]
%
\textsl{Samenstelling promotiecommissie:}\\[0.5\baselineskip]
%
\begin{minipage}[b]{0.34\textwidth}
Rector Magnificus,\\
\loop
  \ifnum \value{@membersprinted} < \value{@members}
    \addtocounter{@membersprinted}{1}
    \ifcase\value{@membersprinted}
      \typeout{Thesis style--internal error printing members.}\\ \or
        \@membera\\ \or
        \@memberb\\ \or
        \@memberc\\ \or
        \@memberd\\ \or
        \@membere\\ \or
        \@memberf\\ \or
        \@memberg\\ \or
        \@memberh\\ \or
        \@memberi\\ \or
  \else \typeout{Thesis style--internal error printing members.}\\
  \fi
\repeat
\end{minipage}
%
\hspace{0.2cm}
\begin{minipage}[b]{0.64\textwidth}
voorzitter\\
\loop
  \ifnum \value{@affiliationsprinted} < \value{@affiliations}
    \addtocounter{@affiliationsprinted}{1}
    \ifcase\value{@affiliationsprinted}
      \typeout{Thesis style--internal error printing member affiliations.} \or
        \@affiliationa\\ \or
        \@affiliationb\\ \or
        \@affiliationc\\ \or
        \@affiliationd\\ \or
        \@affiliatione\\ \or
        \@affiliationf\\ \or
        \@affiliationg\\ \or
        \@affiliationh\\ \or
        \@affiliationi\\
  \else \typeout{Thesis style--internal error printing affiliations.}
  \fi
\repeat
\end{minipage}

\vspace*{\fill}

\includegraphics{logos_for_titlepage}
\\[0.5\baselineskip]

Copyright $\copyright$ 2014 by \@firstname~\@lastname~\\[0.5\baselineskip]

All rights reserved. No part of this book may be reproduced, stored
in a retrieval system, or transmitted, in any form or by any means,
without prior permission from the copyright owner.\\[0.5\baselineskip]

ISBN \@isbnno\\[0.5\baselineskip]
Casimir PhD Series Delft-Leiden 2013-36\\[0.5\baselineskip]
Cover design: L.D. Swakman\\[0.5\baselineskip]
Printed by Gildeprint Drukkerijen --- www.gildeprint.nl\\[0.5\baselineskip]
An electronic version of this thesis is available at www.library.tudelft.nl/dissertations\\[0.5\baselineskip]



%
%Keywords: \@keywords \\[0.5\baselineskip]

%%% IOS PRESS stuff, use only if printing with them  %%%

%
%\textsl{Published and distributed by IOS Press under the imprint Delft University Press}
%
%\begin{tabbing}
%Publisher \& Distributor \qqquad \= Distributor in the USA and Canada\\
%IOS Press \> IOS Press, Inc.\\
%Nieuwe Hemweg 6b \> 4502 Rachael Manor Drive\\
%1013 BG Amsterdam \>  Fairfax, VA 22032\\
%Netherlands \> USA\\
%fax:+31-20-687 0019 \> fax: +1-703-323 3668\\
%email: info@iospress.nl \> e-mail: sales@iospress.com
%\end{tabbing}
%%
%LEGAL NOTICE \\
%The publisher is not responsible for the use which might be made
%of the following information.\\[0.3\baselineskip]
\noindent \@work 
%\vspace*{\fill}\\ [1.25\baselineskip]
%
% PRINTED IN THE NETHERLANDS\\[0.5\baselineskip]
%


\providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Copyright}{bm:Copyright}
\clearpage
}


%% ========================== Dedication page ============================= %%

\newcommand{\dedicationpage}{
\thispagestyle{empty}
\qquad
\vspace{50pt}

{\Large \textsl{
\hspace*{\fill}
\@dedicationtop \\ [0.25\baselineskip]
\hspace*{\fill}
\@dedicationmiddle \\ [0.25\baselineskip]
\hspace*{\fill}
\@dedicationbottom
}}

\vspace*{\fill}
\noindent \@support
\providecommand\pdfbookmark[3][]{} \pdfbookmark[0]{Dedication}{bm:Dedication}
\clearpage
}

%% ========================== Additional files ============================ %%

\input{details}

\endinput

%% ======================================================================== %%