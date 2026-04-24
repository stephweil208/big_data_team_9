import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import re
import sys

st.set_page_config(
    page_title="InsideInsight",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0f1117; }
[data-testid="stSidebar"] { background-color: #161b27; border-right: 0.5px solid #2a2f3e; }
[data-testid="stHeader"] { background-color: #0f1117; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
h1, h2, h3 { color: #e0e4f0 !important; }
p, div, span, label { color: #e0e4f0; }
.kpi-card { background: #161b27; border: 0.5px solid #2a2f3e; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px; }
.kpi-label { font-size: 11px; color: #8890a4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 600; color: #e0e4f0; }
.kpi-sub { font-size: 11px; color: #8890a4; margin-top: 3px; }
.insight-card { background: #161b27; border: 0.5px solid #2a2f3e; border-left: 3px solid #5DCAA5; border-radius: 10px; padding: 16px 20px; margin: 12px 0; }
.insight-title { font-size: 14px; font-weight: 600; color: #e0e4f0; margin-bottom: 4px; }
.insight-sub { font-size: 12px; color: #8890a4; }
.score-leader { background: #1a2e1f; border: 0.5px solid #3B6D11; border-radius: 8px; padding: 6px 14px; color: #97C459; font-size: 13px; font-weight: 600; display: inline-block; }
.score-mid { background: #2e2210; border: 0.5px solid #854F0B; border-radius: 8px; padding: 6px 14px; color: #EF9F27; font-size: 13px; font-weight: 600; display: inline-block; }
.score-low { background: #2e1010; border: 0.5px solid #A32D2D; border-radius: 8px; padding: 6px 14px; color: #F09595; font-size: 13px; font-weight: 600; display: inline-block; }
.section-header { font-size: 11px; font-weight: 600; color: #8890a4; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 0.5px solid #2a2f3e; }
.ai-rec-card { background: #161b27; border: 0.5px solid #2a2f3e; border-left: 3px solid #5DCAA5; border-radius: 10px; padding: 16px 20px; margin: 8px 0; }
.ai-rec-title { font-size: 13px; font-weight: 600; color: #5DCAA5; margin-bottom: 6px; }
.ai-rec-body { font-size: 12px; color: #8890a4; line-height: 1.6; white-space: pre-wrap; }
.traffic-green { background: #1a2e1f; border: 0.5px solid #3B6D11; border-radius: 10px; padding: 16px; text-align: center; }
.traffic-amber { background: #2e2210; border: 0.5px solid #854F0B; border-radius: 10px; padding: 16px; text-align: center; }
.traffic-red { background: #2e1010; border: 0.5px solid #A32D2D; border-radius: 10px; padding: 16px; text-align: center; }
.traffic-label { font-size: 11px; color: #8890a4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.traffic-status { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
.traffic-detail { font-size: 11px; color: #8890a4; }
</style>
""", unsafe_allow_html=True)

ROOM_TYPE_MAP = {
    'entire home/apt': 'Entire home',
    'private room': 'Private room',
    'shared room': 'Shared room',
    'hotel room': 'Hotel room',
}

MONTH_MAP = {
    0: 'Unknown', 1: 'January', 2: 'February', 3: 'March',
    4: 'April', 5: 'May', 6: 'June', 7: 'July',
    8: 'August', 9: 'September', 10: 'October',
    11: 'November', 12: 'December'
}

def safe_int(val, default=0):
    try:
        if pd.isna(val): return default
        return int(float(val))
    except: return default

def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except: return default

@st.cache_data
def load_features():
    path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_features.csv')
    df = pd.read_csv(path)
    if 'room_type' in df.columns:
        df['room_type_clean'] = df['room_type'].str.lower().map(ROOM_TYPE_MAP).fillna(df['room_type'])
    else:
        df['room_type'] = 'entire home/apt'
        df['room_type_clean'] = 'Entire home'
    return df

@st.cache_data
def load_listings():
    path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_listings.csv')
    return pd.read_csv(path)

@st.cache_data
def load_nlp():
    path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_nlp.csv')
    return pd.read_csv(path)

@st.cache_data
def load_reviews():
    path = os.path.join(os.path.dirname(__file__), 'sample_data', 'sample_reviews.csv')
    return pd.read_csv(path)

features   = load_features()
listings   = load_listings()
nlp_df     = load_nlp()
reviews_df = load_reviews()

# ── sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### insideinsight")
    st.markdown("<p style='color:#8890a4;font-size:12px;margin-top:-8px;'>Airbnb host intelligence</p>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate",
                    ["Dashboard", "Pricing simulator", "Health check", "AI advisor"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown("<p class='section-header'>Find your listing</p>", unsafe_allow_html=True)

    url_input = st.text_input("Paste Airbnb URL",
                               placeholder="airbnb.com/rooms/755701",
                               label_visibility="collapsed")
    listing_id_from_url = None
    if url_input:
        match = re.search(r'(\d{5,})', url_input)
        if match:
            listing_id_from_url = int(match.group(1))
            st.success(f"ID detected: {listing_id_from_url}")
        else:
            st.warning("No listing ID found in URL")

    st.markdown("<p class='section-header' style='margin-top:12px;'>Or filter manually</p>",
                unsafe_allow_html=True)

    city = st.selectbox("City", sorted(features['city'].unique()))
    city_features = features[features['city'] == city].copy()
    city_listings = listings[listings['city'] == city].copy()

    neighbourhoods = sorted(city_features['neighborhood'].dropna().unique())
    selected_neighbourhood = st.selectbox("Neighbourhood", ["All"] + neighbourhoods)

    room_types_raw   = sorted(city_features['room_type'].dropna().unique())
    room_types_clean = [ROOM_TYPE_MAP.get(str(r).lower(), r) for r in room_types_raw]
    selected_room_display = st.selectbox("Room type", ["All"] + room_types_clean)
    selected_room_raw = None
    if selected_room_display != "All":
        reverse_map = {v.lower(): k for k, v in ROOM_TYPE_MAP.items()}
        selected_room_raw = reverse_map.get(selected_room_display.lower(), selected_room_display)

    st.markdown("<p class='section-header' style='margin-top:12px;'>Optional filters</p>",
                unsafe_allow_html=True)
    price_range    = st.slider("Price range ($/night)", 0, 1000, (0, 500), step=10)
    min_beds       = st.selectbox("Min bedrooms", ["Any", 1, 2, 3, 4])
    superhost_only = st.checkbox("Superhost listings only")
    high_occ_only  = st.checkbox("High occupancy only (>60%)")
    instant_book   = st.checkbox("Instant bookable only")

    st.divider()
    st.markdown("<p style='color:#8890a4;font-size:10px;'>MSBA 6331 · Carlson School<br>University of Minnesota</p>",
                unsafe_allow_html=True)

# ── filters ───────────────────────────────────────────────────────────
filtered = city_features.copy()
if selected_neighbourhood != "All":
    filtered = filtered[filtered['neighborhood'] == selected_neighbourhood]
if selected_room_raw:
    filtered = filtered[filtered['room_type'].str.lower() == selected_room_raw.lower()]
filtered = filtered[
    (filtered['price'] >= price_range[0]) &
    (filtered['price'] <= price_range[1])
]
if high_occ_only:
    filtered = filtered[filtered['occupancy_rate'] >= 0.6]

listing_cols = ['listing_id','accommodates','bedrooms','number_of_reviews',
                'review_scores_rating','review_scores_location','review_scores_value',
                'host_is_superhost','instant_bookable',
                'estimated_occupancy_l365d','estimated_revenue_l365d','amenities_count']
available_cols = [c for c in listing_cols if c in city_listings.columns]
merged = filtered.merge(city_listings[available_cols], on='listing_id', how='left')

if superhost_only and 'host_is_superhost' in merged.columns:
    merged = merged[merged['host_is_superhost'] == 't']
if instant_book and 'instant_bookable' in merged.columns:
    merged = merged[merged['instant_bookable'] == 't']
if min_beds != "Any" and 'bedrooms' in merged.columns:
    merged = merged[merged['bedrooms'] >= int(min_beds)]

if len(merged) == 0:
    st.warning("No listings match your filters. Try adjusting the filters in the sidebar.")
    st.stop()

if listing_id_from_url and listing_id_from_url in merged['listing_id'].values:
    selected_id = listing_id_from_url
else:
    selected_id = st.selectbox(
        f"Select listing ({len(merged)} found)",
        merged['listing_id'].tolist(),
        format_func=lambda x: f"#{x} — {merged[merged['listing_id']==x]['neighborhood'].values[0]}"
    )

row = merged[merged['listing_id'] == selected_id].iloc[0]

listing_price  = safe_float(row['price'])
median_price   = safe_float(row['median_neighborhood_price'])
price_gap      = safe_float(row['price_gap_pct'])
occ_rate       = safe_float(row['occupancy_rate'])
comp_score     = safe_float(row['competitive_score'])
peak_month     = safe_int(row['peak_month'])
amenity_score  = safe_float(row['amenity_score'])
occ_days       = safe_float(row.get('estimated_occupancy_l365d'), occ_rate * 365)
ann_revenue    = safe_float(row.get('estimated_revenue_l365d'), listing_price * occ_days)
price_diff     = abs(listing_price - median_price)
revenue_upside = round(price_diff * occ_days, 0) if price_gap < 0 else 0
room_clean     = ROOM_TYPE_MAP.get(str(row['room_type']).lower(), str(row['room_type']))
neighbourhood  = str(row['neighborhood'])
peak_name      = MONTH_MAP.get(peak_month, 'Unknown')

nbhd_avg_occ  = city_features[city_features['neighborhood'] == neighbourhood]['occupancy_rate'].mean()
effective_occ = occ_days if occ_days > 0 else round(safe_float(nbhd_avg_occ) * 365, 0)

nlp_row  = nlp_df[nlp_df['listing_id'] == selected_id]
has_nlp  = len(nlp_row) > 0
if has_nlp:
    nlp           = nlp_row.iloc[0]
    sentiment     = safe_float(nlp['avg_sentiment_score'])
    pct_pos       = safe_float(nlp['pct_positive'])
    total_rev     = safe_int(nlp['total_reviews'])
    sent_cat      = str(nlp['sentiment_category'])
    top_praise    = str(nlp['top_praise']).replace('_',' ').title()
    top_complaint = str(nlp['top_complaint']).replace('_',' ').title()
    sent_color    = "#97C459" if sentiment > 0.5 else "#EF9F27" if sentiment > 0 else "#F09595"

def score_badge(score):
    if score >= 80:
        return f"<span class='score-leader'>Market leader · {score:.0f}/100</span>"
    elif score >= 40:
        return f"<span class='score-mid'>Mid-tier · {score:.0f}/100</span>"
    else:
        return f"<span class='score-low'>Fixer-upper · {score:.0f}/100</span>"

def traffic_card(col, label, status, detail, color):
    css = {'green': 'traffic-green', 'amber': 'traffic-amber', 'red': 'traffic-red'}[color]
    text_color = {'green': '#97C459', 'amber': '#EF9F27', 'red': '#F09595'}[color]
    col.markdown(f"""<div class='{css}'>
        <div class='traffic-label'>{label}</div>
        <div class='traffic-status' style='color:{text_color};'>{status}</div>
        <div class='traffic-detail'>{detail}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown(f"### {room_clean} · {neighbourhood}, {city.title()}")
    st.markdown(
        f"Listing #{selected_id} &nbsp;·&nbsp; "
        f"{safe_int(row.get('accommodates'))} guests &nbsp;·&nbsp; "
        f"{safe_int(row.get('bedrooms'))} beds &nbsp;·&nbsp; "
        f"{safe_int(row.get('amenities_count'))} amenities &nbsp;·&nbsp; "
        f"{safe_int(row.get('number_of_reviews'))} reviews",
        unsafe_allow_html=True
    )
    st.markdown(score_badge(comp_score), unsafe_allow_html=True)
    st.divider()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Nightly price</div>
            <div class='kpi-value'>${listing_price:.0f}</div>
            <div class='kpi-sub'>vs ${median_price:.0f} median</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        gap_color = "#F09595" if price_gap < -10 else "#97C459" if price_gap > 10 else "#EF9F27"
        gap_label = "Underpriced" if price_gap < -5 else "Overpriced" if price_gap > 5 else "Near median"
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Price gap</div>
            <div class='kpi-value' style='color:{gap_color};'>{price_gap:+.1f}%</div>
            <div class='kpi-sub'>{gap_label}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Occupancy</div>
            <div class='kpi-value'>{occ_rate*100:.1f}%</div>
            <div class='kpi-sub'>{occ_days:.0f} days / yr</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Annual revenue</div>
            <div class='kpi-value'>${ann_revenue:,.0f}</div>
            <div class='kpi-sub'>Estimated actual</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        upside_color = "#5DCAA5" if revenue_upside > 0 else "#8890a4"
        upside_text  = f"+${revenue_upside:,.0f}" if revenue_upside > 0 else "Optimised"
        upside_sub   = f"${price_diff:.0f}/night × {occ_days:.0f} days" if revenue_upside > 0 else "At or above median"
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Revenue upside</div>
            <div class='kpi-value' style='color:{upside_color};'>{upside_text}</div>
            <div class='kpi-sub'>{upside_sub}</div>
        </div>""", unsafe_allow_html=True)

    if revenue_upside > 0:
        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>You are underpriced by ${price_diff:.0f}/night — costing an estimated ${revenue_upside:,.0f}/year in lost revenue</div>
            <div class='insight-sub'>Formula: Price gap (${price_diff:.0f}) × Occupancy days ({occ_days:.0f}) = ${revenue_upside:,.0f}/year &nbsp;·&nbsp; Peak demand: {peak_name} &nbsp;·&nbsp; Amenity score: {amenity_score:.0f}/100</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>Your pricing is competitive in {neighbourhood}</div>
            <div class='insight-sub'>Priced {abs(price_gap):.1f}% above neighbourhood median &nbsp;·&nbsp; Peak demand: {peak_name} &nbsp;·&nbsp; Monitor occupancy to ensure premium is sustainable</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<p class='section-header'>Pricing position</p>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=['Your listing', 'Neighbourhood median', 'City median'],
            y=[listing_price, median_price, float(city_features['price'].median())],
            marker_color=['#378ADD', '#5DCAA5', '#888780'],
            text=[f'${listing_price:.0f}', f'${median_price:.0f}',
                  f'${city_features["price"].median():.0f}'],
            textposition='outside', textfont=dict(color='#e0e4f0', size=13)
        ))
        fig.update_layout(
            yaxis_title="Price per night ($)", showlegend=False, height=300,
            margin=dict(t=30, b=10, l=10, r=10),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#8890a4'),
            yaxis=dict(gridcolor='#2a2f3e', color='#8890a4'),
            xaxis=dict(color='#8890a4')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<p class='section-header'>Score breakdown</p>", unsafe_allow_html=True)
        scores = {
            'Amenities': amenity_score,
            'Reviews':   safe_float(row.get('review_rank_percentile')),
            'Occupancy': safe_float(row.get('occupancy_rank_percentile')),
        }
        fig2 = go.Figure(go.Bar(
            x=list(scores.values()), y=list(scores.keys()),
            orientation='h', marker_color='#5DCAA5',
            text=[f'{v:.0f}' for v in scores.values()],
            textposition='outside', textfont=dict(color='#e0e4f0', size=12)
        ))
        fig2.update_layout(
            xaxis=dict(range=[0, 120], gridcolor='#2a2f3e', color='#8890a4'),
            yaxis=dict(color='#8890a4'), height=300,
            margin=dict(t=30, b=10, l=10, r=40),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#8890a4')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("<p class='section-header'>Review intelligence</p>", unsafe_allow_html=True)

    if has_nlp:
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Sentiment score</div>
                <div class='kpi-value' style='color:{sent_color};'>{sentiment:.2f}</div>
                <div class='kpi-sub'>{sent_cat} · {pct_pos*100:.0f}% positive</div>
            </div>""", unsafe_allow_html=True)
        with n2:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Reviews analysed</div>
                <div class='kpi-value'>{total_rev}</div>
                <div class='kpi-sub'>Via VADER sentiment</div>
            </div>""", unsafe_allow_html=True)
        with n3:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Top praise</div>
                <div class='kpi-value' style='color:#97C459;font-size:16px;'>{top_praise}</div>
                <div class='kpi-sub'>Most mentioned positive</div>
            </div>""", unsafe_allow_html=True)
        with n4:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>Top complaint</div>
                <div class='kpi-value' style='color:#F09595;font-size:16px;'>{top_complaint}</div>
                <div class='kpi-sub'>Most mentioned issue</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<p class='section-header' style='margin-top:12px;'>Topic mentions</p>",
                    unsafe_allow_html=True)
        topics = {
            'Cleanliness': safe_int(nlp.get('cleanliness_mentions')),
            'Location':    safe_int(nlp.get('location_mentions')),
            'Host':        safe_int(nlp.get('host_mentions')),
            'Check-in':    safe_int(nlp.get('checkin_mentions')),
            'WiFi':        safe_int(nlp.get('wifi_mentions')),
            'Noise':       safe_int(nlp.get('noise_mentions')),
            'Value':       safe_int(nlp.get('value_mentions')),
        }
        max_val = max(topics.values()) if max(topics.values()) > 0 else 1
        fig3 = go.Figure(go.Bar(
            x=list(topics.keys()), y=list(topics.values()),
            marker_color=['#97C459' if v == max_val else '#5DCAA5' for v in topics.values()],
            text=list(topics.values()), textposition='outside',
            textfont=dict(color='#e0e4f0', size=11)
        ))
        fig3.update_layout(
            yaxis_title="Mentions", showlegend=False, height=250,
            margin=dict(t=20, b=10, l=10, r=10),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#8890a4'),
            yaxis=dict(gridcolor='#2a2f3e', color='#8890a4'),
            xaxis=dict(color='#8890a4')
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        def score_display(val, label):
            v = safe_float(val)
            if v > 0:
                return f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{v:.2f}<span style='font-size:13px;color:#8890a4;'> / 5</span></div></div>"
            return f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value' style='color:#8890a4;'>N/A</div></div>"
        r1, r2, r3 = st.columns(3)
        r1.markdown(score_display(row.get('review_scores_rating'), 'Overall rating'), unsafe_allow_html=True)
        r2.markdown(score_display(row.get('review_scores_location'), 'Location score'), unsafe_allow_html=True)
        r3.markdown(score_display(row.get('review_scores_value'), 'Value score'), unsafe_allow_html=True)
        st.caption("NLP analysis not available for this listing.")

    listing_reviews = reviews_df[reviews_df['listing_id'] == selected_id]
    if len(listing_reviews) > 0:
        st.divider()
        st.markdown("<p class='section-header'>Recent guest reviews</p>", unsafe_allow_html=True)
        for _, rev in listing_reviews.head(3).iterrows():
            comment = str(rev.get('comments', ''))[:300]
            date    = str(rev.get('review_date', ''))[:10]
            name    = str(rev.get('reviewer_name', 'Guest'))
            st.markdown(f"""<div class='insight-card'>
                <div class='insight-title'>{name} &nbsp;·&nbsp; <span style='color:#8890a4;font-weight:400;font-size:12px;'>{date}</span></div>
                <div class='insight-sub'>{comment}{'...' if len(str(rev.get('comments',''))) > 300 else ''}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PRICING SIMULATOR
# ══════════════════════════════════════════════════════════════════════
elif page == "Pricing simulator":
    st.markdown("### Pricing simulator")
    st.markdown(f"*Test different price points for listing #{selected_id} · {room_clean} · {neighbourhood}*")
    st.divider()

    st.markdown("<p class='section-header'>Adjust your nightly price</p>", unsafe_allow_html=True)

    min_price = max(10, int(listing_price * 0.4))
    max_price = int(listing_price * 2.5)
    sim_price = st.slider(
        "Simulated nightly price ($)",
        min_value=min_price, max_value=max_price,
        value=int(listing_price), step=5
    )

    sim_gap       = ((sim_price - median_price) / median_price * 100) if median_price > 0 else 0
    sim_revenue   = round(sim_price * effective_occ, 0)
    current_rev   = round(listing_price * effective_occ, 0)
    sim_vs_actual = sim_revenue - current_rev
    is_nbhd_avg   = occ_days == 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Simulated price</div>
            <div class='kpi-value'>${sim_price}</div>
            <div class='kpi-sub'>Current: ${listing_price:.0f}/night</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        gap_col = "#F09595" if sim_gap < -10 else "#97C459" if sim_gap > 10 else "#EF9F27"
        gap_lbl = "Underpriced" if sim_gap < -5 else "Overpriced" if sim_gap > 5 else "At median"
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>vs neighbourhood median</div>
            <div class='kpi-value' style='color:{gap_col};'>{sim_gap:+.1f}%</div>
            <div class='kpi-sub'>{gap_lbl} · median ${median_price:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        rev_col = "#97C459" if sim_vs_actual > 0 else "#F09595" if sim_vs_actual < 0 else "#8890a4"
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Projected annual revenue</div>
            <div class='kpi-value'>${sim_revenue:,.0f}</div>
            <div class='kpi-sub' style='color:{rev_col};'>{sim_vs_actual:+,.0f} vs current</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        occ_risk = "No change" if sim_gap <= 10 else "May reduce occupancy" if sim_gap > 25 else "Monitor closely"
        occ_col  = "#8890a4" if sim_gap <= 10 else "#F09595" if sim_gap > 25 else "#EF9F27"
        occ_note = " (neighbourhood avg)" if is_nbhd_avg else ""
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Occupancy basis</div>
            <div class='kpi-value' style='color:{occ_col};font-size:15px;'>{occ_risk}</div>
            <div class='kpi-sub'>{effective_occ:.0f} days/yr{occ_note}</div>
        </div>""", unsafe_allow_html=True)

    if sim_vs_actual > 0:
        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>At ${sim_price}/night you would earn ${sim_vs_actual:,.0f} more per year</div>
            <div class='insight-sub'>Formula: ${sim_price} × {effective_occ:.0f} days = ${sim_revenue:,.0f} &nbsp;·&nbsp; vs current ${listing_price:.0f} × {effective_occ:.0f} days = ${current_rev:,.0f} &nbsp;·&nbsp; difference = +${sim_vs_actual:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    elif sim_vs_actual < 0:
        st.markdown(f"""<div class='insight-card' style='border-left-color:#F09595;'>
            <div class='insight-title'>At ${sim_price}/night you would earn ${abs(sim_vs_actual):,.0f} less per year</div>
            <div class='insight-sub'>Formula: ${sim_price} × {effective_occ:.0f} days = ${sim_revenue:,.0f} &nbsp;·&nbsp; vs current ${listing_price:.0f} × {effective_occ:.0f} days = ${current_rev:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>This is your current price — move the slider to simulate scenarios</div>
            <div class='insight-sub'>Occupancy basis: {effective_occ:.0f} days/yr{occ_note if is_nbhd_avg else ''}</div>
        </div>""", unsafe_allow_html=True)

    if is_nbhd_avg:
        st.caption(f"Note: This listing has no booking history. Using {neighbourhood} neighbourhood average of {effective_occ:.0f} days/year.")

    st.divider()
    st.markdown("<p class='section-header'>Revenue curve — price vs annual revenue</p>",
                unsafe_allow_html=True)

    price_range_sim = list(range(min_price, max_price + 1, 5))
    revenue_curve   = [p * effective_occ for p in price_range_sim]

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=price_range_sim, y=revenue_curve,
        mode='lines', line=dict(color='#5DCAA5', width=2)
    ))
    fig_sim.add_vline(x=sim_price, line_color='#378ADD', line_width=2, line_dash='dash')
    fig_sim.add_vline(x=median_price, line_color='#EF9F27', line_width=1, line_dash='dot')
    fig_sim.add_annotation(x=sim_price, y=max(revenue_curve)*0.95,
                           text=f"Simulated: ${sim_price}",
                           font=dict(color='#378ADD', size=11), showarrow=False)
    fig_sim.add_annotation(x=median_price, y=max(revenue_curve)*0.85,
                           text=f"Median: ${median_price:.0f}",
                           font=dict(color='#EF9F27', size=11), showarrow=False)
    fig_sim.update_layout(
        xaxis_title="Nightly price ($)", yaxis_title="Projected annual revenue ($)",
        height=350, showlegend=False,
        margin=dict(t=20, b=40, l=60, r=20),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#8890a4'),
        yaxis=dict(gridcolor='#2a2f3e', color='#8890a4', tickformat='$,.0f'),
        xaxis=dict(gridcolor='#2a2f3e', color='#8890a4', tickformat='$,.0f')
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    st.markdown("<p class='section-header'>Neighbourhood price distribution</p>",
                unsafe_allow_html=True)
    nbhd_prices = city_features[city_features['neighborhood'] == neighbourhood]['price'].dropna()
    if len(nbhd_prices) > 0:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=nbhd_prices, nbinsx=20,
            marker_color='#5DCAA5', opacity=0.7
        ))
        fig_hist.add_vline(x=sim_price, line_color='#378ADD', line_width=2, line_dash='dash',
                           annotation_text=f"Simulated: ${sim_price}",
                           annotation_font_color='#378ADD')
        fig_hist.add_vline(x=median_price, line_color='#EF9F27', line_width=2, line_dash='dot',
                           annotation_text=f"Median: ${median_price:.0f}",
                           annotation_font_color='#EF9F27')
        fig_hist.update_layout(
            xaxis_title="Nightly price ($)", yaxis_title="Number of listings",
            height=280, showlegend=False,
            margin=dict(t=30, b=40, l=40, r=20),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='#8890a4'),
            yaxis=dict(gridcolor='#2a2f3e', color='#8890a4'),
            xaxis=dict(gridcolor='#2a2f3e', color='#8890a4')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════
elif page == "Health check":
    st.markdown("### Listing health check")
    st.markdown(f"*{room_clean} · {neighbourhood}, {city.title()} · #{selected_id}*")
    st.divider()

    col_score, col_context = st.columns([1, 3])
    with col_score:
        st.markdown("<p class='section-header'>Competitive score</p>", unsafe_allow_html=True)
        st.markdown(score_badge(comp_score), unsafe_allow_html=True)
        st.markdown(f"<p style='color:#8890a4;font-size:11px;margin-top:6px;'>vs all {neighbourhood} listings</p>",
                    unsafe_allow_html=True)
    with col_context:
        st.markdown("<p class='section-header'>What this means</p>", unsafe_allow_html=True)
        if comp_score >= 80:
            meaning = f"Your listing ranks in the top tier for {neighbourhood}. Strong pricing, high occupancy, and good reviews. Focus on maintaining performance and testing small price increases."
        elif comp_score >= 40:
            meaning = f"Your listing is mid-tier in {neighbourhood}. Specific improvements are available — check the scorecard below to see which dimensions are holding your score back."
        else:
            meaning = f"Your listing needs attention in {neighbourhood}. The scorecard below identifies the highest-priority fixes. Addressing pricing and amenities typically has the fastest impact."
        st.markdown(f"<p style='color:#8890a4;font-size:13px;line-height:1.6;'>{meaning}</p>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("<p class='section-header'>5-dimension health scorecard</p>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    if price_gap < -20:
        p_status, p_detail, p_color = "Underpriced", f"{abs(price_gap):.0f}% below median", "red"
    elif price_gap < -5:
        p_status, p_detail, p_color = "Slightly low", f"{abs(price_gap):.0f}% below median", "amber"
    elif price_gap <= 15:
        p_status, p_detail, p_color = "Well priced", f"{price_gap:+.0f}% vs median", "green"
    else:
        p_status, p_detail, p_color = "Overpriced", f"{price_gap:.0f}% above median", "amber"
    traffic_card(c1, "Pricing", p_status, p_detail, p_color)

    occ_pct = occ_rate * 100
    if occ_pct >= 60:
        o_status, o_detail, o_color = "High demand", f"{occ_pct:.0f}% occupancy", "green"
    elif occ_pct >= 30:
        o_status, o_detail, o_color = "Moderate", f"{occ_pct:.0f}% occupancy", "amber"
    elif occ_days == 0:
        o_status, o_detail, o_color = "No history", "New listing", "amber"
    else:
        o_status, o_detail, o_color = "Low demand", f"{occ_pct:.0f}% occupancy", "red"
    traffic_card(c2, "Occupancy", o_status, o_detail, o_color)

    if amenity_score >= 70:
        a_status, a_detail, a_color = "Well equipped", f"{amenity_score:.0f}/100", "green"
    elif amenity_score >= 40:
        a_status, a_detail, a_color = "Average", f"{amenity_score:.0f}/100", "amber"
    else:
        a_status, a_detail, a_color = "Needs work", f"{amenity_score:.0f}/100", "red"
    traffic_card(c3, "Amenities", a_status, a_detail, a_color)

    rev_rank  = safe_float(row.get('review_rank_percentile'))
    n_reviews = safe_int(row.get('number_of_reviews'))
    if rev_rank >= 70:
        r_status, r_detail, r_color = "Strong", f"Top {100-rev_rank:.0f}% · {n_reviews} reviews", "green"
    elif rev_rank >= 30:
        r_status, r_detail, r_color = "Average", f"{rev_rank:.0f}th pctile · {n_reviews} reviews", "amber"
    elif n_reviews == 0:
        r_status, r_detail, r_color = "No reviews", "Build history", "amber"
    else:
        r_status, r_detail, r_color = "Below average", f"{rev_rank:.0f}th pctile", "red"
    traffic_card(c4, "Reviews", r_status, r_detail, r_color)

    if has_nlp:
        if sentiment >= 0.6:
            s_status, s_detail, s_color = sent_cat, f"{sentiment:.2f} · {pct_pos*100:.0f}% positive", "green"
        elif sentiment >= 0.2:
            s_status, s_detail, s_color = sent_cat, f"{sentiment:.2f} · {pct_pos*100:.0f}% positive", "amber"
        else:
            s_status, s_detail, s_color = sent_cat, f"{sentiment:.2f} · {pct_pos*100:.0f}% positive", "red"
    else:
        s_status, s_detail, s_color = "No data", "Needs reviews", "amber"
    traffic_card(c5, "Sentiment", s_status, s_detail, s_color)

    st.divider()
    st.markdown("<p class='section-header'>Improvement priorities</p>", unsafe_allow_html=True)

    priorities = []
    if p_color == 'red':
        priorities.append(("Pricing — high priority",
                           f"You are {abs(price_gap):.0f}% below the {neighbourhood} median. Raising your price by ${price_diff:.0f}/night could recover ${revenue_upside:,.0f}/year.",
                           "#F09595"))
    if a_color == 'red':
        priorities.append(("Amenities — high priority",
                           f"Your amenity score of {amenity_score:.0f}/100 is below average. Add WiFi, coffee machine, self check-in, and dedicated workspace.",
                           "#F09595"))
    if p_color == 'amber':
        priorities.append(("Pricing — worth reviewing",
                           f"You are {abs(price_gap):.0f}% below median. A gradual price increase of ${price_diff*0.5:.0f}/night is a low-risk starting point.",
                           "#EF9F27"))
    if a_color == 'amber':
        priorities.append(("Amenities — room to improve",
                           f"Score {amenity_score:.0f}/100. Adding 3-5 targeted amenities could push you into the top quartile.",
                           "#EF9F27"))
    if has_nlp and top_complaint:
        priorities.append(("Guest feedback — action needed",
                           f"Top complaint is {top_complaint}. Address this in your house rules or welcome message to improve future ratings.",
                           "#EF9F27"))
    if peak_month > 0:
        priorities.append(("Seasonal opportunity",
                           f"Your neighbourhood peaks in {peak_name}. Raise rates 15-25% for 6 weeks around peak to capture demand.",
                           "#5DCAA5"))

    if not priorities:
        st.markdown(f"""<div class='insight-card'>
            <div class='insight-title'>Your listing is in good health</div>
            <div class='insight-sub'>No critical issues detected. Focus on maintaining performance and monitoring seasonal demand.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for title, detail, col in priorities:
            st.markdown(f"""<div class='insight-card' style='border-left-color:{col};'>
                <div class='insight-title' style='color:{col};'>{title}</div>
                <div class='insight-sub'>{detail}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p class='section-header'>How you compare — neighbourhood leaderboard</p>",
                unsafe_allow_html=True)

    nbhd_df = city_features[city_features['neighborhood'] == neighbourhood].copy()
    top10   = nbhd_df.nlargest(10, 'competitive_score')[
        ['listing_id','room_type','price','competitive_score',
         'amenity_score','occupancy_rank_percentile']
    ].copy()
    top10['is_yours'] = top10['listing_id'] == selected_id
    top10['label']    = top10['listing_id'].apply(lambda x: f"...{str(x)[-6:]}")
    top10.loc[top10['is_yours'], 'label'] = top10.loc[top10['is_yours'], 'listing_id'].apply(
        lambda x: f"YOU ...{str(x)[-6:]}"
    )

    fig_lb = go.Figure(go.Bar(
        x=top10['competitive_score'],
        y=top10['label'],
        orientation='h',
        marker_color=['#378ADD' if y else '#5DCAA5' for y in top10['is_yours']],
        text=[f"{s:.0f}" for s in top10['competitive_score']],
        textposition='outside', textfont=dict(color='#e0e4f0', size=11)
    ))
    fig_lb.update_layout(
        xaxis=dict(range=[0, 115], gridcolor='#2a2f3e', color='#8890a4', title='Competitive score'),
        yaxis=dict(color='#8890a4', autorange='reversed'),
        height=360,
        margin=dict(t=20, b=20, l=110, r=40),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='#8890a4')
    )
    st.plotly_chart(fig_lb, use_container_width=True)
    st.caption("Blue = your listing · Green = others in neighbourhood · Higher score = more competitive")

# ══════════════════════════════════════════════════════════════════════
# AI ADVISOR
# ══════════════════════════════════════════════════════════════════════
elif page == "AI advisor":
    st.markdown("### AI action plan")
    st.markdown(f"*Listing #{selected_id} · {room_clean} · {neighbourhood}, {city.title()}*")
    st.divider()

    nlp_context = ""
    if has_nlp:
        nlp_context = f"""
Sentiment score     : {sentiment:.2f} ({sent_cat})
Pct positive        : {pct_pos*100:.0f}%
Top praise          : {top_praise}
Top complaint       : {top_complaint}
Reviews analysed    : {total_rev}"""

    context_block = f"""
Nightly price       : ${listing_price:.0f}  (neighbourhood median: ${median_price:.0f})
Price gap           : {price_gap:+.1f}%  ({'underpriced' if price_gap < 0 else 'overpriced'})
Occupancy           : {occ_rate*100:.1f}%  ({occ_days:.0f} days/year)
Annual revenue      : ${ann_revenue:,.0f}
Revenue upside      : ${revenue_upside:,.0f}
Competitive score   : {comp_score:.0f}/100
Amenity score       : {amenity_score:.0f}/100
Peak demand month   : {peak_name}
Room type           : {room_clean}
Neighbourhood       : {neighbourhood}, {city.title()}{nlp_context}
    """

    st.markdown("<p class='section-header'>Listing data used for recommendations</p>",
                unsafe_allow_html=True)
    st.code(context_block, language=None)

    use_live_agent = st.toggle("Use live AI agent (LangGraph + Groq LLaMA 3.3)", value=False)

    if use_live_agent:
        st.markdown("""<div class='insight-card'>
            <div class='insight-title'>Live agent mode active</div>
            <div class='insight-sub'>Pipeline: retrieve context → LLaMA 3.3 70B analysis → grounded recommendations · Powered by LangGraph + Groq · First run builds FAISS index (~30s)</div>
        </div>""", unsafe_allow_html=True)

    if st.button("Generate AI action plan", type="primary", use_container_width=True):
        if use_live_agent:
            try:
                with st.spinner("Initialising LangGraph agent — building FAISS index..."):
                    sys.path.insert(0, os.path.dirname(__file__))
                    from agent import initialize_agent, action_plan as agent_action_plan
                    features_path = os.path.join(os.path.dirname(__file__),
                                                  'sample_data', 'sample_features.csv')
                    nlp_path = os.path.join(os.path.dirname(__file__),
                                             'sample_data', 'sample_nlp.csv')
                    initialize_agent(features_path=features_path, nlp_path=nlp_path)

                with st.spinner("Running agent — retrieve → analyse → recommend..."):
                    result = agent_action_plan(str(selected_id))

                st.markdown(f"""<div class='ai-rec-card'>
                    <div class='ai-rec-title'>LangGraph agent — LLaMA 3.3 70B via Groq</div>
                    <div class='ai-rec-body'>{result}</div>
                </div>""", unsafe_allow_html=True)
                st.caption("Generated by LangGraph multi-agent pipeline: retrieve → analyse → recommend. All numbers grounded in Spark-computed statistics.")

            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.warning("Falling back to grounded recommendations.")
                use_live_agent = False

        if not use_live_agent:
            with st.spinner("Analysing your listing..."):
                if price_gap < -10:
                    pricing_rec = f"Raise your nightly rate by ${price_diff:.0f} to match the {neighbourhood} median of ${median_price:.0f}. At your current occupancy of {occ_days:.0f} days/year this recovers ${revenue_upside:,.0f} annually.\n\nSource stats: price_gap_pct={price_gap:.1f}%, estimated_occupancy={occ_days:.0f} days, revenue_upside=${revenue_upside:,.0f}"
                elif price_gap > 15:
                    pricing_rec = f"Your price of ${listing_price:.0f} is {price_gap:.1f}% above the {neighbourhood} median of ${median_price:.0f}. If occupancy drops below 40% consider a reduction of ${price_diff*0.5:.0f}/night.\n\nSource stats: price_gap_pct=+{price_gap:.1f}%, occupancy_rate={occ_rate*100:.1f}%"
                else:
                    pricing_rec = f"Your pricing of ${listing_price:.0f} is well-positioned in {neighbourhood} ({price_gap:+.1f}% vs median). Focus on review quality and amenity improvements.\n\nSource stats: price_gap_pct={price_gap:.1f}%, competitive_score={comp_score:.0f}/100"

                if amenity_score < 40:
                    amenity_rec = f"Your amenity score of {amenity_score:.0f}/100 is significantly below the neighbourhood benchmark. Prioritise: fast WiFi, coffee machine, self check-in, dedicated workspace.\n\nSource stats: amenity_score={amenity_score:.0f}/100"
                elif amenity_score < 70:
                    amenity_rec = f"Your amenity score of {amenity_score:.0f}/100 is moderate. Adding 3-5 targeted amenities could push you into the top quartile for {neighbourhood}.\n\nSource stats: amenity_score={amenity_score:.0f}/100"
                else:
                    amenity_rec = f"Your amenity score of {amenity_score:.0f}/100 is competitive. Highlight your top amenities prominently in your listing title and first photo.\n\nSource stats: amenity_score={amenity_score:.0f}/100"

                if peak_month > 0:
                    season_rec = f"Your neighbourhood peaks in {peak_name}. Implement dynamic pricing: raise your rate 15-25% starting 6 weeks before {peak_name}.\n\nSource stats: peak_month={peak_name}, peak_occupancy_rate={safe_float(row.get('peak_occupancy_rate'))*100:.1f}%"
                else:
                    season_rec = "Seasonal demand data is not yet available for this listing. Recommendations will appear once 90+ days of booking history accumulates."

                if has_nlp:
                    nlp_rec = f"Your guests most frequently praise {top_praise} — lead with this in your listing title. Your top complaint is {top_complaint}: address this directly in your house rules.\n\nSource stats: top_praise={top_praise}, top_complaint={top_complaint}, sentiment={sentiment:.2f} ({sent_cat}), {pct_pos*100:.0f}% positive"
                else:
                    nlp_rec = "Review sentiment analysis is not yet available for this listing. Ensure you have at least 5 reviews before NLP analysis can run."

                for title, body in [
                    ("1. Pricing recommendation", pricing_rec),
                    ("2. Amenity recommendation",  amenity_rec),
                    ("3. Seasonal recommendation", season_rec),
                    ("4. Review intelligence",      nlp_rec),
                ]:
                    st.markdown(f"""<div class='ai-rec-card'>
                        <div class='ai-rec-title'>{title}</div>
                        <div class='ai-rec-body'>{body}</div>
                    </div>""", unsafe_allow_html=True)

                st.caption("Recommendations cite computed statistics from your listing data. Toggle 'Use live AI agent' for LangGraph + Groq powered recommendations.")