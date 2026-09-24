# TRAC-Phish Revision 8 — Case Studies

Best model: **LogReg(C=4.0)** — local explanations = top signed Linear-SHAP char n-grams.


## [GB] TP — score 1.0

- URL: `http://login.apk-mfacebook.ml/login.php`
- y_true=1 y_pred=1
- top SHAP n-grams: .ml/(+1.3162); .ml(+1.0061); ml/(+0.5493); /log(+0.5091); apk(-0.5036); gin.(+0.4626); .apk(-0.4481); ogin.(+0.4324)
- lexical: url_len=39; n_digits=0; n_susp_tokens=3; digit_ratio=0; char_entropy=4.1; n_subdomains=2; n_hyphens=1; tld_suspicious=1

## [GB] TP — score 1.0

- URL: `https://asvimed.it/frd/Office36555555/Validation`
- y_true=1 y_pred=1
- top SHAP n-grams: 5(+0.4356); /o(+0.4153); d/(+0.3996); valid(+0.3448); asv(+0.3197); /of(+0.3084); v(+0.3034); 5555(+0.2879)
- lexical: url_len=48; n_digits=8; n_susp_tokens=0; digit_ratio=0.167; char_entropy=4.23; n_subdomains=1; n_hyphens=0; tld_suspicious=0

## [GB] FP — score 1.0

- URL: `https://etsy.app.link/z6151dAHTX`
- y_true=0 y_pred=1
- top SHAP n-grams: k/(+0.4480); .app(+0.4360); .link(+0.4338); pp(+0.3781); link(+0.3681); p.l(+0.3613); .app.(+0.3246); z6(+0.2989)
- lexical: url_len=32; n_digits=4; n_susp_tokens=0; digit_ratio=0.125; char_entropy=4.37; n_subdomains=2; n_hyphens=0; tld_suspicious=1

## [GB] FP — score 1.0

- URL: `https://www.netangels.ru/out/?url=Z0FBQUFBQmJfU0pJUFExN3BCNXI3YUQ4VkdVZGVsM1BGRjFZUmVWMGZkQjR1SWw4SGc3Mm4yLWFpZ2UzWTNLX0p4SkJpZHlyVndGV0xmdmxYSzNlMjd1ek5iNTBRcWVlSTMxbzZWcGg2dS04WU`
- y_true=0 y_pred=1
- top SHAP n-grams: fb(+0.4458); x(+0.3594); q(+0.3556); v(+0.3062); bg(+0.2450); z(+0.2324); /o(+0.2283); fbg(+0.2128)
- lexical: url_len=194; n_digits=24; n_susp_tokens=0; digit_ratio=0.124; char_entropy=5.57; n_subdomains=1; n_hyphens=0; tld_suspicious=0

## [GB] FN — score 0.0

- URL: `http://www.baseball.deals/2018/06/06/2018-top-5-youth-usa-aluminum-bat-review/`
- y_true=1 y_pred=0
- top SHAP n-grams: 201(-0.4633); /20(-0.4474); /06/(-0.3221); /2(-0.3174); /201(-0.2948); als/(-0.2592); -(-0.2487); w/(+0.2353)
- lexical: url_len=78; n_digits=13; n_susp_tokens=0; digit_ratio=0.167; char_entropy=4.54; n_subdomains=1; n_hyphens=7; tld_suspicious=0

## [GB] FN — score 0.0

- URL: `http://www.fst-dynamo-kyiv.org/news/230-chempionat-kmo-fst-dinamo-ukrajini-z-legkoatletichnogo-krosu-2018/`
- y_true=1 y_pred=0
- top SHAP n-grams: /news(-0.4365); -k(-0.3523); news(-0.3122); o-(-0.2937); ews(-0.2825); -(-0.2822); 201(-0.2550); fst(-0.2321)
- lexical: url_len=106; n_digits=7; n_susp_tokens=0; digit_ratio=0.066; char_entropy=4.67; n_subdomains=1; n_hyphens=11; tld_suspicious=0

## [GB] TN — score 0.0

- URL: `http://www.cuisine-etudiante.fr/page/2/`
- y_true=0 y_pred=0
- top SHAP n-grams: ge/2(-0.5825); page/(-0.5698); age/2(-0.5410); ge/2/(-0.4557); e/2/(-0.4309); e/2(-0.3552); /2(-0.3148); e-(-0.2311)
- lexical: url_len=39; n_digits=1; n_susp_tokens=0; digit_ratio=0.0256; char_entropy=4.08; n_subdomains=1; n_hyphens=1; tld_suspicious=0

## [GB] TN — score 0.0

- URL: `http://www.glowkitchen.com/2015/05/spring-broccoli-peas-coconut-curry/#more-11805`
- y_true=0 y_pred=0
- top SHAP n-grams: #(-0.5828); /#(-0.3764); 201(-0.2766); /20(-0.2670); -(-0.2186); 5(+0.2161); /2(-0.1873); .glo(-0.1810)
- lexical: url_len=81; n_digits=11; n_susp_tokens=0; digit_ratio=0.136; char_entropy=4.6; n_subdomains=1; n_hyphens=5; tld_suspicious=0

## [PP2026] FP-transfer — score 1.0

- URL: `www.paypal.com/pay/billing`
- y_true=0 y_pred=1
- top SHAP n-grams: pay(+1.2737); bill(+0.6429); ypal(+0.5161); payp(+0.5140); paypa(+0.5114); aypal(+0.4839); aypa(+0.4565); billi(+0.4331)
- lexical: url_len=26; n_digits=0; n_susp_tokens=2; digit_ratio=0; char_entropy=3.66; n_subdomains=1; n_hyphens=0; tld_suspicious=0

## [PP2026] FP-transfer — score 1.0

- URL: `www.amazon.com/dp/b07mbc77kk/ref`
- y_true=0 y_pred=1
- top SHAP n-grams: maz(+0.4945); k/(+0.4708); azon(+0.4513); mbc(+0.4301); mazon(+0.4214); mazo(+0.4107); p/(+0.4024); dp/(+0.3672)
- lexical: url_len=32; n_digits=4; n_susp_tokens=1; digit_ratio=0.125; char_entropy=4.03; n_subdomains=1; n_hyphens=0; tld_suspicious=0

## [PP2026] FN-transfer — score 0.0

- URL: `https://www.xwcbblog.com`
- y_true=1 y_pred=0
- top SHAP n-grams: blog(-1.2618); xw(-0.7512); blo(-0.6098); blog.(-0.5093); s://w(-0.5081); b(+0.4860); x(+0.4502); w.x(-0.3993)
- lexical: url_len=24; n_digits=0; n_susp_tokens=0; digit_ratio=0; char_entropy=3.75; n_subdomains=1; n_hyphens=0; tld_suspicious=0

## [PP2026] FN-transfer — score 0.0

- URL: `https://www.c-and-a-sk.sk`
- y_true=1 y_pred=0
- top SHAP n-grams: -(-0.5214); -a-(-0.4784); a-(-0.4507); -and-(-0.4411); -and(-0.4328); s://w(-0.4324); k(+0.4306); -s(-0.3963)
- lexical: url_len=25; n_digits=0; n_susp_tokens=0; digit_ratio=0; char_entropy=3.67; n_subdomains=1; n_hyphens=3; tld_suspicious=0