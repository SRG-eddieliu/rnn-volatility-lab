"""Build the corrected portfolio PDFs from the public aggregate results file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak

INK='#243037'; TEAL='#176B72'; PINK='#AB4569'; GRAY='#627078'; WIDTH=504


def fonts():
    directory=Path('/System/Library/Fonts/Supplemental')
    regular=directory/'Arial.ttf'; bold=directory/'Arial Bold.ttf'
    if not regular.exists():
        regular=Path(font_manager.findfont('DejaVu Sans'))
        bold=Path(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold')))
    pdfmetrics.registerFont(TTFont('Body',str(regular)))
    pdfmetrics.registerFont(TTFont('BodyBold',str(bold)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    font_manager.fontManager.addfont(str(regular))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(regular)).get_name(),
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#A0A8AD','text.color':INK,'axes.labelcolor':INK,
                         'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':'white','savefig.facecolor':'white'})


STYLES={
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=29,leading=34,textColor=colors.HexColor(INK),spaceAfter=16),
 'subtitle':ParagraphStyle('subtitle',fontName='Body',fontSize=14,leading=20,textColor=colors.HexColor(GRAY),spaceAfter=14),
 'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=21,leading=26,textColor=colors.HexColor(INK),spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=12,leading=16,textColor=colors.HexColor(TEAL),spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.4,leading=15.1,textColor=colors.HexColor(INK),spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.7,leading=12.3,textColor=colors.HexColor(GRAY),spaceAfter=8),
 'table':ParagraphStyle('table',fontName='Body',fontSize=8.7,leading=12,textColor=colors.HexColor(INK)),
 'head':ParagraphStyle('head',fontName='BodyBold',fontSize=8.7,leading=12,textColor=colors.HexColor(INK)),
 'eq':ParagraphStyle('eq',fontName='Body',fontSize=11,leading=17,textColor=colors.HexColor(INK),leftIndent=10,spaceBefore=5,spaceAfter=12),
 'kicker':ParagraphStyle('kicker',fontName='BodyBold',fontSize=9,leading=12,textColor=colors.HexColor(TEAL),spaceAfter=12),
}


def p(text,style='body'): return Paragraph(text,STYLES[style])


def table(rows,widths,compact=False):
    t=Table([[p(str(cell),'head' if i==0 else 'table') for cell in row] for i,row in enumerate(rows)],
            colWidths=widths,hAlign='LEFT',repeatRows=1)
    pad=4 if compact else 7
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                          ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),pad),
                          ('BOTTOMPADDING',(0,0),(-1,-1),pad),
                          ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F1')),
                          ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9CACB0')),
                          ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DCE1E3'))]))
    return t


def footer(label):
    def draw(c,doc):
        c.setStrokeColor(colors.HexColor('#D4DCDF'));c.setLineWidth(.5);c.line(54,48,558,48)
        c.setFillColor(colors.HexColor(GRAY));c.setFont('Body',8)
        c.drawString(54,33,label+' | Eddie Liu | September 2026');c.drawRightString(558,33,str(doc.page))
    return draw


def pdf(path,title,story,count):
    doc=SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=65,
                          title=title,author='Eddie Liu',subject='Corrected historical walk-forward experiment; single primary seed',
                          creator='Eddie Liu - research portfolio')
    doc.build(story,onFirstPage=footer(title),onLaterPages=footer(title))
    actual=len(PdfReader(path).pages)
    if actual!=count: raise ValueError(f'{path.name}: expected {count} pages, got {actual}')


def build(results:Path,out:Path):
    fonts();out.mkdir(parents=True,exist_ok=True)
    d=json.loads(results.read_text());metrics=d['metrics'];bykey={r['model']:r for r in metrics}
    neural=metrics[3:]; best_nn=min(neural,key=lambda r:r['qlike']);best_all=min(metrics,key=lambda r:r['qlike'])
    best_mse=min(metrics,key=lambda r:r['mse']);base=bykey['garch_t'];n=d['prepared']['test_observations']
    floors=sum(r['floor_count_1e12'] for r in metrics)
    paired=[r for r in d['paired_loss_intervals'] if r['loss']=='qlike']
    best_ci=next(r for r in paired if r['model']==best_nn['model'])
    source='Source: corrected-v1 aggregate results; cached ^GSPC snapshot; no fresh market-data download.'
    provenance=('The underlying project originated in collaborative research. This revised portfolio report is '
                'presented by Eddie Liu. Contributor names and course identifiers are omitted; the byline does '
                'not imply sole authorship of the original experiments.')
    temp=tempfile.TemporaryDirectory(prefix='volatility-report-');figdir=Path(temp.name)
    def save(fig,name):
        fig.savefig(figdir/name,dpi=200,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    def image(name,height): return Image(str(figdir/name),width=WIDTH,height=height)

    fig,ax=plt.subplots(figsize=(7.7,4.0))
    for i,r in enumerate(paired):
        point=r['difference_vs_garch'];lo=r['ci95_lower'];hi=r['ci95_upper']
        ax.hlines(i,lo,hi,color=TEAL,linewidth=1.5);ax.plot(point,i,'o',color=TEAL,markersize=5)
    ax.set_yticks(range(len(paired)),[bykey[r['model']]['label'] for r in paired]);ax.invert_yaxis()
    ax.axvline(0,color=GRAY,linewidth=.8);ax.grid(axis='x',color='#E4E9EB',linewidth=.5)
    ax.set_xlabel('Mean QLIKE difference versus GARCH-t (negative favors candidate)')
    save(fig,'intervals.png')

    fig,ax=plt.subplots(figsize=(7.7,3.7))
    ratios=[r['forecast_to_proxy_mean_ratio'] for r in metrics]
    ax.barh([r['label'] for r in metrics],ratios,color=TEAL,height=.6);ax.invert_yaxis()
    ax.axvline(1,color=GRAY,linestyle='--',linewidth=1)
    ax.set_xlim(0,max(1.25,max(ratios)*1.2))
    for i,value in enumerate(ratios):ax.text(value+.015,i,f'{value:.3f}',va='center',fontsize=9)
    ax.set_xlabel('Mean predicted variance / mean squared-return proxy (1 = equal means)')
    ax.grid(axis='x',color='#E4E9EB',linewidth=.5);ax.set_axisbelow(True)
    save(fig,'calibration.png')

    order=['input_gate','candidate_gate','output_gate','forget_gate']
    gates={g['gate']:g for g in d['gate_stats'] if g['model']=='pure_lstm'}
    fig,ax=plt.subplots(figsize=(7.7,3.3));ys=np.arange(4)
    ax.scatter([gates[g]['pearson'] for g in order],ys-.09,color=TEAL,label='Pearson',s=40)
    ax.scatter([gates[g]['spearman'] for g in order],ys+.09,facecolor='white',edgecolor=PINK,marker='s',label='Spearman',s=40)
    ax.set_yticks(ys,[g.replace('_gate','').capitalize() for g in order]);ax.invert_yaxis()
    ax.set_xlim(-1,1);ax.set_xticks(np.arange(-1,1.01,.25));ax.axvline(0,color=GRAY,linewidth=.8)
    ax.grid(axis='x',color='#E4E9EB',linewidth=.5);ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel(f'Correlation with same-date VIX close ({n:,} dates)')
    save(fig,'gates.png')

    metric_rows=[['Forecast','MSE / 10^-7','QLIKE','Mean h / mean y']]
    for r in metrics:metric_rows.append([r['label'],f"{r['mse']/1e-7:.5f}",f"{r['qlike']:.5f}",f"{r['forecast_to_proxy_mean_ratio']:.3f}"])
    periods=['2010-2016','2017-2019','2020-2021','2022-2024']
    calendar={(r['model'],r['period']):r for r in d['calendar_periods']}
    calendar_rows=[['Forecast',*periods]]
    for r in metrics:calendar_rows.append([r['label'],*[f"{calendar[r['model'],period]['qlike']:.3f}" for period in periods]])
    sign='lower' if best_nn['qlike']<base['qlike'] else 'higher'
    result_sentence=(f"{best_all['label']} has the lowest QLIKE point estimate ({best_all['qlike']:.4f}); "
                     f"{best_mse['label']} has the lowest MSE ({best_mse['mse']:.3e}). "
                     f"The best neural QLIKE point estimate is {best_nn['label']} ({best_nn['qlike']:.4f}), "
                     f"{sign} than GARCH-t ({base['qlike']:.4f}).")

    main=[p('CORRECTED EXPERIMENT / SEPTEMBER 24, 2026','kicker'),
          p('Volatility Forecasting:<br/>GARCH and Recurrent Models','title'),
          p('A controlled walk-forward comparison<br/>with calibration and uncertainty diagnostics','subtitle'),
          p('<b>Eddie Liu</b>'),Spacer(1,5),
          table([['EXPERIMENT SCOPE'],[f'Nine specifications, {n:,} common test dates, February 5, 2010 to December 11, 2024. Six neural models each trained across 178 expanding-window folds. One primary training seed.']], [WIDTH]),
          p('Research question','h2'),
          p('Can recurrent models improve one-day equity-index variance forecasts over parsimonious '
            'classical baselines? The comparison separates implementation correctness, forecast positivity, '
            'mean calibration, and loss-based performance.'),
          p('What changed','h2'),
          p('The rerun repairs daily GARCH state updates, corrects a target inverse mismatch, aligns '
            'neural training availability, seeds before model construction, and evaluates train/validation '
            'loss at the same restored checkpoint. Positive log-ratio hybrids replace the original '
            'unconstrained additive residual specification.'),
          p('Main finding','h2'),p(result_sentence),
          p(f'Across the nine complete prediction series, {floors} forecasts are at or below 1e-12. '
            'Positivity does not establish calibration or superiority. The study uses a previously '
            'examined historical snapshot, not an untouched prospective holdout.'),
          p(source,'small'),PageBreak()]

    main += [p('01 / A Comparable Experiment','h1'),
             table([['Element','Corrected design'],
                    ['Data and target','Cached daily ^GSPC prices, 2003-2024. Target y is squared log return in daily decimal-return variance units.'],
                    ['Classical baselines','Student-t GARCH(1,1); lagged mean of 21 squared returns; EWMA with lambda 0.94.'],
                    ['Neural inputs','21 rows ending at t-1: log return, squared return, absolute return, annualized 21-day rolling volatility. Feature/ratio hybrids also include lagged GARCH variance.'],
                    ['Common availability','Modeling frame begins after the 756-return GARCH warmup. Initial neural training span has 777 rows, yielding 756 sequences; validation has 252 observations.'],
                    ['Walk-forward tests','178 non-overlapping blocks of 21 observations. Training expands; validation and model weights are reselected per fold. The incomplete trailing block is excluded.'],
                    ['Neural settings','8 units; dropout 0.1; Adam 0.001; batch 64; maximum 35 epochs; patience 6; seed 42 plus split ID. CPU execution; fixed-length recurrence unrolled.']], [115,389]),
             p('Forecast timing','h2'),
             p('A forecast for date t is issued using information through t-1. Feature and target '
               'transform statistics are fitted only on the training sample. The validation sample '
               'selects early stopping. Later folds may train on observations that were test data '
               'in an earlier fold, once those observations become historical.'),
             p('Why the evaluation now begins in 2010','h2'),
             p('The original hybrid runs had very little early training history after GARCH warmup. '
               'The rerun requires a full shared neural training span and therefore starts testing '
               'later. Old and new loss levels must not be interpreted as a one-change before/after '
               'comparison. GARCH retains its own expanding return-estimation history; equal neural '
               'dates do not make every estimator\'s effective memory identical.'),
             p('Squared returns are a noisy second-moment proxy. Their interpretation as conditional '
               'variance additionally assumes a zero conditional mean. The 2008 crisis is in training '
               'history, not this corrected test period.','small'),PageBreak()]

    main += [p('02 / Results on Identical Test Dates','h1'),
             p(f'All rows use the same {n:,} targets and dates. Three classical baselines and six '
               'freshly trained neural specifications are included. This is a new historical '
               'walk-forward run, not a rescore of the old forecasts.'),table(metric_rows,[142,110,110,142]),
             p('MSE = mean[(y - h)<super>2</super>]<br/>QLIKE = mean[log(h) + y/h]','eq'),
             p('Both losses are lower-is-better. QLIKE uses a fixed 1e-12 numerical floor for y '
               'and h; no floor was selected against test losses. Negative QLIKE levels are normal '
               'under this convention. The MSE column is scaled by 10^-7; these are not returns '
               'or percentages.'),
             p('Interpretation','h2'),p(result_sentence),
             p('Mean h / mean y is a coarse level-calibration diagnostic, not a full conditional '
               'calibration test. A value below one means the average variance prediction is below '
               'the average realized proxy. A good average ratio does not rule out tail or regime failures.'),
             p('Both original-scale MSE and QLIKE can be robust to an unbiased noisy proxy under '
               'suitable conditions [2]. Different loss weighting and finite-sample behavior can '
               'produce different rankings. Point-estimate selection across several candidates '
               'does not establish statistically reliable superiority.'),PageBreak()]

    main += [p('03 / Paired Loss Uncertainty','h1'),
             image('intervals.png',280),
             p('Figure 1. Candidate minus GARCH-t mean QLIKE, with 95% pointwise percentile intervals '
               'from a paired stationary bootstrap. Average block length 21; 2,000 replications; '
               'seed 20260924. Negative values favor the candidate. All comparisons use 3,738 dates.','small'),
             p('What the interval addresses','h2'),
             p('Daily candidate and benchmark losses are paired by date. Resampling blocks retains '
               'some serial dependence rather than treating daily observations as independent. '
               'The interval concerns the mean loss difference conditional on these trained forecasts '
               'and suitable time-series assumptions.'),
             p(f"For {best_nn['label']}, the observed QLIKE difference versus GARCH is "
               f"<b>{best_ci['difference_vs_garch']:+.4f}</b>, with interval "
               f"<b>[{best_ci['ci95_lower']:+.4f}, {best_ci['ci95_upper']:+.4f}]</b>. "
               'This model is highlighted because it has the lowest neural QLIKE point estimate; '
               'that selection is post hoc and is not accounted for by the individual interval.'),
             p('What it does not address','h2'),
             p('These are not simultaneous intervals across all candidates and do not quantify '
               'neural initialization uncertainty, data-vendor variation, research selection, or '
               'future regime shifts. Block-length sensitivity at 5 and 63 is retained in the '
               'aggregate evidence file. A single historical comparison is not proof of deployment value.'),PageBreak()]

    main += [p('04 / Positivity Is Not Calibration','h1'),
             image('calibration.png',245),
             p('Figure 2. Mean predicted variance divided by mean squared-return proxy on the '
               'common test sample. The dashed line marks equal means; it is not a significance threshold.','small'),
             p('A positive hybrid construction','h2'),
             p('The ratio hybrid learns z = log(max(y, 1e-8)/g), where g is GARCH variance, and '
               'reconstructs h = g exp(predicted z). Finite log predictions and positive g yield '
               'positive variance; numeric overflow or underflow is rejected. The target floor '
               'prevents log(0), and is distinct from an output floor.'),
             p('The objective still matters','h2'),
             p('The pure/feature models train on standardized log targets; ratio models train '
               'on standardized log ratios. Squared error on log y targets E[log y | information]. '
               'Exponentiating does not generally recover E[y | information]. Correcting the '
               'inverse transform therefore fixes a bug, not this estimand mismatch.'),
             p('Under a conditional Gaussian illustration without clipping, '
               'exp(E[log(r<super>2</super>)]) is approximately 0.281 times the conditional variance. '
               'This explains why well-fitted log-target models can underforecast arithmetic '
               'variance. It is not an empirical correction factor for this dataset.'),
             p('A next controlled study should compare a positive output trained directly under '
               'QLIKE or another original-scale objective. No such additional model is claimed '
               'as completed here.','small'),PageBreak()]

    main += [p('05 / Calendar-Period Diagnostics','h1'),
             p('The following calendar slices were fixed before inspecting the new test losses. '
               'They are descriptive partitions, not a fitted regime classification. Each column '
               'compares models on the same dates within that period.'),
             table(calendar_rows,[144,90,90,90,90]),
             p('Table 2. Mean QLIKE by period; lower is better. Observation counts: '+
               '; '.join(f"{period}: {calendar['garch_t',period]['n']:,}" for period in periods)+'.','small'),
             p('Why inspect subperiods?','h2'),
             p('An aggregate mean can hide unstable relative performance or a handful of large '
               'loss contributions. Calendar diagnostics help identify questions for follow-up, '
               'but choosing favorable periods after seeing them would be a selection problem.'),
             p('What remains untested','h2'),
             p('The main run uses one initialization schedule. It does not establish seed '
               'robustness, performance across assets, conditional calibration, or stability '
               'under a different target proxy. The historical dataset has been studied before; '
               'this is not an untouched final holdout.'),
             p('Trading relevance','h2'),
             p('Variance forecasts may inform risk sizing, volatility targeting, or risk estimates. '
               'Those are potential applications, not results of this study. No strategy, '
               'execution model, transaction-cost analysis, portfolio P&amp;L, or Sharpe-ratio '
               'improvement is claimed.'),
             p('The appropriate conclusion is specific to the forecast target, information set, '
               'sample, objective, and baselines actually tested. More complex architecture is '
               'not itself evidence of better research or better investment performance.'),PageBreak()]

    main += [p('06 / Recurrent Gates as Diagnostics','h1'),
             image('gates.png',240),
             p('Figure 3. Corrected-run pure-LSTM activations, averaged across units and lags 1-20, '
               'versus same-date VIX close. Pearson and rank correlations use 3,738 matched dates. '
               'The appendix lists all 21 activation/variant summaries.','small'),
             p('What is observable','h2'),
             p(f"The pure-LSTM forget-gate aggregate has Pearson correlation "
               f"<b>{gates['forget_gate']['pearson']:+.3f}</b> and Spearman correlation "
               f"<b>{gates['forget_gate']['spearman']:+.3f}</b> with VIX in this rerun. "
               'These are descriptive associations, not identified mechanisms.'),
             p('Three interpretation limits','h2'),
             p('<b>Timing:</b> same-date VIX close is ex-post context and was not a predictor '
               'available at the previous-day forecast origin.<br/>'
               '<b>Pooling:</b> the series combines separately trained models across folds; '
               'unit identities and fitted behavior need not remain stable.<br/>'
               '<b>Identification:</b> an average activation does not identify a memory horizon, '
               'causal response, or tradable signal.'),
             p('Stronger evidence would require controlled ablations, within-fold checks, '
               'multiple seeds, and correctly lagged predictive tests. Candidate activations '
               'are shown with the gate summaries for consistency; they are not sigmoid gates.'),PageBreak()]

    main += [p('07 / Conclusions and Provenance','h1'),
             p('The corrected run demonstrates a reproducible research workflow with explicit '
               'forecast timing, shared neural sample availability, positive hybrid construction, '
               'simple baselines, and paired temporal uncertainty. The numerical and timing '
               'defects identified in the previous report have regression coverage.'),p(result_sentence),
             p('Next research priorities','h2'),
             p('First, align a positive variance output with an original-scale proper loss. '
               'Second, repeat across seeds and assets with a genuinely held-out evaluation '
               'plan. Third, connect forecasts to a prespecified economic objective only if '
               'the statistical evidence justifies the added complexity.'),
             p('References','h2'),
             p('[1] Bollerslev (1986), Generalized autoregressive conditional heteroskedasticity. '
               '<link href="https://doi.org/10.1016/0304-4076(86)90063-1" color="'+TEAL+'">Journal of Econometrics.</link><br/>'
               '[2] Patton (2011), Volatility forecast comparison using imperfect volatility proxies. '
               '<link href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf" color="'+TEAL+'">Journal of Econometrics.</link><br/>'
               '[3] Hochreiter and Schmidhuber (1997), Long Short-Term Memory. '
               '<link href="https://doi.org/10.1162/neco.1997.9.8.1735" color="'+TEAL+'">Neural Computation.</link><br/>'
               '[4] Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder. '
               '<link href="https://arxiv.org/abs/1406.1078" color="'+TEAL+'">arXiv:1406.1078.</link><br/>'
               '[5] <link href="https://arch.readthedocs.io/en/latest/univariate/forecasting.html" color="'+TEAL+'">arch forecasting documentation</link> and '
               '<link href="https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html" color="'+TEAL+'">stationary bootstrap implementation.</link>','small'),
             p('Research record','h2'),
             p('The public repository includes code, tests, pinned research dependencies, this '
               'report builder, and aggregate results with input/code/forecast hashes. Raw market '
               'data and complete prediction CSVs remain local. The previous report is preserved '
               'in Git history; corrected-v1 is a separate experiment, not a silent overwrite '
               'of historical scores.','small'),
             p(provenance,'small'),
             p('<link href="https://github.com/SRG-eddieliu/rnn-volatility-lab" color="'+TEAL+'">github.com/SRG-eddieliu/rnn-volatility-lab</link>','small')]
    pdf(out/'volatility-research.pdf','Volatility Forecasting - Corrected Study',main,8)

    garch=d['garch_diagnostics']
    diagnostic=[p('CORRECTED EXPERIMENT / TECHNICAL APPENDIX','kicker'),
                p('Volatility Forecasting:<br/>Diagnostic Appendix','title'),p('<b>Eddie Liu</b>'),
                p('A / Verification Summary','h2'),
                table([['Check','Observed evidence'],
                       ['Run coverage',f'6 neural specifications x 178 folds = 1,068 fits. Nine complete forecast series, each with {n:,} identical dates and targets.'],
                       ['GARCH fitting',f"{garch['refits']} parameter refits; all convergence flags are zero. {garch['same_as_previous']} / {garch['transitions']} consecutive forecasts repeat exactly."],
                       ['Forecast validity',f'All final forecasts are finite and positive; {floors} at or below 1e-12 across all nine series.'],
                       ['Regression suite','Tests cover the actual arch fixed-parameter reference, new-return response, forecast timing, transforms, input validity, sequences, bootstrap mechanics, and CPU training/resumption.'],
                       ['Metric verification','Independent array calculations reconcile with the repository MSE and QLIKE functions. All model CSV hashes match their completion records.']], [130,374]),
                p('Repaired daily recursion','h2'),
                p('h<sub>t</sub> = omega + alpha r<sub>t-1</sub><super>2</super> + beta h<sub>t-1</sub>','eq'),
                p('On refit dates, parameters and endpoint variance come from an expanding fit '
                  'using returns through t-1. Between refits, parameters stay fixed while every '
                  'new return updates the state. Internal percent-squared variance is divided '
                  'by 10,000 for the output units.'),
                p('The corrected experiment does not certify upstream price history against '
                  'an independent vendor, every unused legacy notebook, or production controls. '
                  'A converged fit is a numerical condition, not proof that the model is correct.','small'),PageBreak()]

    train_rows=[['Neural model','Mean epochs','Train loss','Validation loss','Paired gap']]
    for key,log in d['training_diagnostics'].items():
        train_rows.append([bykey[key]['label'],f"{log['mean_epochs']:.1f}",f"{log['mean_checkpoint_train_loss']:.4f}",
                           f"{log['mean_checkpoint_val_loss']:.4f}",f"{log['mean_checkpoint_gap']:+.4f}"])
    diagnostic += [p('B / Training and Numerical Checks','h1'),table(train_rows,[136,80,88,100,100]),
                   p('Table A1. Means across 178 folds. Train and validation losses are evaluated '
                     'in inference mode at the same restored best-validation checkpoint. The '
                     'losses are standardized-target MSE, not original-scale variance MSE.','small'),
                   p('Interpretation','h2'),
                   p('The corrected paired gap fixes the old practice of subtracting independent '
                     'minima from different epochs. A gap is still a diagnostic, not a theorem '
                     'about overfitting. Pure/feature log targets and ratio targets differ, so '
                     'cross-family gap magnitudes are not directly comparable.'),
                   p('Transform and reconstruction','h2'),
                   p('For log targets, the forward map uses log(max(y, 1e-8)); the inverse '
                     'exponentiates without subtracting epsilon. Clipping is not invertible '
                     'below its floor, and the tests explicitly check this boundary behavior.'),
                   p('For ratio hybrids, the signed log-ratio target is standardized without '
                     'taking another logarithm. Reconstruction uses exp(log(g) + predicted z) '
                     'in float64 and rejects non-finite or nonpositive numerical results.'),
                   p('Reproducibility and interruption handling','h2'),
                   p('Seeds are set before model construction. TensorFlow deterministic CPU '
                     'operations are enabled. A run signature binds input bytes, relevant code, '
                     'configuration, and split selection. An output directory with a conflicting '
                     'signature is rejected. A split is complete only when predictions, required '
                     'gate rows, and the final training log agree on coverage.'),
                   p('Deterministic smoke tests reproduce within the tested environment. This '
                     'is not a guarantee of bitwise portability to another TensorFlow version '
                     'or hardware platform.','small'),PageBreak()]

    diagnostic += [p('C / Paired Loss-Difference Intervals','h1')]
    for loss in ['mse','qlike']:
        rows=[['Candidate vs GARCH','Difference','95% lower','95% upper']]
        for r in d['paired_loss_intervals']:
            if r['loss']!=loss:continue
            fmt=(lambda x:f'{x/1e-7:+.5f}') if loss=='mse' else (lambda x:f'{x:+.5f}')
            rows.append([bykey[r['model']]['label'],fmt(r['difference_vs_garch']),fmt(r['ci95_lower']),fmt(r['ci95_upper'])])
        diagnostic += [p('MSE differences / 10^-7' if loss=='mse' else 'QLIKE differences','h2'),table(rows,[180,108,108,108],compact=True)]
    diagnostic += [p('Each interval is pointwise, conditional on the trained forecast series, and '
                     'not adjusted for comparing several candidates. Paired stationary bootstrap; '
                     'mean block length 21; 2,000 replications. The full aggregate JSON includes '
                     '5- and 63-observation block sensitivities with 1,000 replications each. '
                     'Negative values favor the candidate.','small'),PageBreak()]

    gate_rows=[['Model','Activation','n','Pearson','Spearman']]
    for g in d['gate_stats']:
        gate_rows.append([bykey[g['model']]['label'],g['gate'].replace('_gate',''),str(g['n']),
                          f"{g['pearson']:+.3f}",f"{g['spearman']:+.3f}"])
    diagnostic += [p('D / Complete Activation Summary','h1'),
                   p('Each row uses the corrected common test period and same-date VIX. Values '
                     'are averaged across units and lags 1-20, then pooled across rolling folds.','small'),
                   table(gate_rows,[144,100,64,98,98],compact=True),
                   p('These are ex-post correlations without interval estimates or a causal '
                     'interpretation. Candidate activations are not sigmoid gates. Raw gate '
                     'outputs, forecasts, and training logs are retained locally for inspection.','small'),PageBreak()]

    env_rows=[['Package','Recorded version']]+[[k,v] for k,v in d['environment'].items()]
    diagnostic += [p('E / Reproducibility and Evidence','h1'),table(env_rows,[250,254]),
                   p('Rerun commands','h2'),
                   p('Install requirements-research.txt in an isolated environment.<br/>'
                     '1. Run scripts/run_corrected_experiment.py with --raw, --out, and --prepare.<br/>'
                     '2. Run scripts/run_all_models.py with the same --out and a bounded --workers value.<br/>'
                     '3. Run scripts/evaluate_corrected_experiment.py with --out, --destination, and the optional VIX snapshot.<br/>'
                     '4. Build the PDFs with reports/build_corrected_report.py from the aggregate results JSON.'),
                   p('Artifact identification','h2'),
                   p('The aggregate results record the raw snapshot SHA-256, relevant source-code '
                     'hashes, configuration, model prediction hashes, and matched VIX snapshot hash. '
                     'The public repository omits raw market data and prediction CSVs. Hashes '
                     'identify the local evidence but do not replace access to it for an independent full rerun.'),
                   p('Limits carried into the conclusions','h2'),
                   p('One primary seed; one equity index; a cached historical dataset already '
                     'examined in earlier research; a noisy daily proxy; log-objective mismatch; '
                     'pointwise bootstrap uncertainty; no economic strategy or deployment test. '
                     'The calendar slices and benchmark checks do not remove these limitations.'),
                   p(provenance,'small'),
                   p('Aggregate evidence: reports/corrected_results.json. Source workflow and '
                     'interpretation boundaries: docs/corrected-experiment-plan.md and '
                     'docs/validation-status.md.','small')]
    pdf(out/'volatility-diagnostics.pdf','Volatility Forecasting - Corrected Appendix',diagnostic,5)
    merged=PdfWriter();merged.append(out/'volatility-research.pdf',outline_item='Corrected Research Study')
    merged.append(out/'volatility-diagnostics.pdf',outline_item='Diagnostic Appendix')
    merged.add_metadata({'/Title':'Volatility Forecasting - Corrected Study and Appendix','/Author':'Eddie Liu',
                         '/Subject':'Corrected historical walk-forward study; single primary seed'})
    with (out/'volatility-research-with-appendix.pdf').open('wb') as stream:merged.write(stream)
    temp.cleanup()
    print(json.dumps({'main_pages':8,'appendix_pages':5,'combined_pages':13,'output':str(out)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).with_name('corrected_results.json'))
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    a=parser.parse_args();build(a.results,a.output_dir)
