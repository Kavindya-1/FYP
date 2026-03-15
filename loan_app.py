import streamlit as st
import joblib
import pandas as pd
import time

xgb_model = joblib.load("credit_model.pkl")
kproto_model = joblib.load("kproto_cluster_model.pkl")
eligible_customers = joblib.load("eligible_customers.pkl")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: linear-gradient(160deg, #0C447C 0%, #042C53 100%); min-height: 100vh; }

.welcome-card {
    background: rgba(255,255,255,0.06);
    border: 0.5px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    max-width: 420px;
    margin: 4rem auto;
    backdrop-filter: blur(4px);
}
.bank-name {
    font-size: 12px; letter-spacing: 3px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase; margin-bottom: 1.2rem;
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 30px; color: white;
    line-height: 1.2; margin-bottom: 0.5rem;
}
.page-subtitle {
    font-size: 14px; color: rgba(255,255,255,0.55);
    margin-bottom: 2rem; line-height: 1.6;
}
.footer-note {
    font-size: 12px; color: rgba(255,255,255,0.3);
    text-align: center; margin-top: 1rem;
}
stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 0.5px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important; color: white !important;
}
.stButton > button {
    width: 100%; background: white !important;
    color: #042C53 !important; border: none !important;
    border-radius: 10px !important; font-weight: 500 !important;
    padding: 0.75rem !important;
}
</style>

<div class="welcome-card">
    <div class="bank-name">National Bank &bull; Loan Portal</div>
    <div class="page-title">Check your loan eligibility</div>
    <p class="page-subtitle">Enter your NIC number to instantly verify your eligibility. Your data is secure and confidential.</p>
</div>
""", unsafe_allow_html=True)

nic = st.text_input("NIC Number", placeholder="e.g. 199012345678")

if st.button("Proceed"):
    if not nic:
        st.error("Please enter your NIC number to proceed.")
    else:
        # Loading game while verifying
        game_placeholder = st.empty()
        game_placeholder.markdown("""
        <div style="background:rgba(255,255,255,0.06);border-radius:16px;padding:1.5rem;text-align:center;color:white;">
            <p style="color:rgba(255,255,255,0.5);font-size:13px;letter-spacing:1px;margin-bottom:1rem">VERIFYING YOUR NIC...</p>
            <p style="font-size:14px;margin-bottom:1rem">While you wait — catch the coins!</p>
            <div id="gameArea" style="position:relative;width:280px;height:140px;background:rgba(255,255,255,0.05);border-radius:10px;overflow:hidden;border:0.5px solid rgba(255,255,255,0.1);margin:0 auto;cursor:none">
                <div id="paddle" style="position:absolute;bottom:6px;width:60px;height:8px;background:white;border-radius:4px;left:110px"></div>
            </div>
            <p style="margin-top:0.75rem;color:rgba(255,255,255,0.6);font-size:13px">Score: <span id="score">0</span></p>
        </div>
        <script>
        let score=0,coins=[],gl,cl;
        const area=document.getElementById('gameArea');
        const paddle=document.getElementById('paddle');
        const scoreEl=document.getElementById('score');
        area.addEventListener('mousemove',e=>{
            const r=area.getBoundingClientRect();
            paddle.style.left=Math.min(220,Math.max(0,e.clientX-r.left-30))+'px';
        });
        function spawn(){
            const c=document.createElement('div');
            c.style.cssText='position:absolute;width:16px;height:16px;border-radius:50%;background:#FAC775;left:'+Math.random()*260+'px;top:-20px';
            area.appendChild(c);coins.push({el:c,y:-20,s:1.5+Math.random()*1.5});
        }
        function tick(){
            const pw=parseInt(paddle.style.left)||110;
            coins.forEach((c,i)=>{
                c.y+=c.s;c.el.style.top=c.y+'px';
                if(c.y>115){if(parseFloat(c.el.style.left)>pw-8&&parseFloat(c.el.style.left)<pw+60){score++;scoreEl.textContent=score;}c.el.remove();coins.splice(i,1);}
            });
        }
        gl=setInterval(tick,16);cl=setInterval(spawn,800);
        </script>
        """, unsafe_allow_html=True)

        time.sleep(3)  # simulate backend check
        game_placeholder.empty()

        # Backend check
        matched = eligible_customers[eligible_customers['MASKED_LEGAL_ID'] == nic]

        if matched.empty:
            st.error("NIC number not found in our records. Please contact your nearest branch.")
        else:
            customer_record = matched.iloc[0]
            if customer_record['Eligibility_Flag'] == 'REJECT':
                st.error("You are not eligible to apply for a loan at this time.")
            else:
                score = customer_record['Internal_Bank_Default_Score']
                band = customer_record['Score_Band']
                if band in ["Very Low Risk", "Low Risk"]:
                    st.success("Congratulations! Your loan application can proceed.")
                elif band == "Medium Risk":
                    st.warning("Your application is under review. A loan officer will contact you.")
                else:
                    st.error("Unfortunately, your loan application cannot be approved at this time.")

st.markdown('<p class="footer-note">Secured &bull; Confidential &bull; Instant results</p>', unsafe_allow_html=True)