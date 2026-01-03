import streamlit as st
import numpy as np
from predict_placement import PlacementPredictor

st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .percentage-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        color: #2ecc71;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .placed {
        background-color: #d4edda;
        color: #155724;
    }
    .not-placed {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

 
st.markdown('<h1 class="main-header">🎓 Student Placement Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

 
if 'predictor' not in st.session_state:
    try:
        with st.spinner("Loading model..."):
            st.session_state.predictor = PlacementPredictor()
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Model file 'placement_prediction_model.pth' not found! Please run the notebook first to train and save the model.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

 
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### About
    This system predicts the probability of student placement based on:
    - **IQ**: Intelligence Quotient
    - **Previous Semester Result**: Academic performance
    - **CGPA**: Cumulative Grade Point Average
    - **Communication Skills**: Score out of 10
    - **Projects Completed**: Number of projects
    
    ### Instructions
    1. Enter student details in the form
    2. Click "Predict Placement" button
    3. View the predicted placement percentage
    
    ### Model Info
    - **Architecture**: Neural Network (5 → 64 → 32 → 16 → 1)
    - **Output**: Placement probability (0-100%)
    """)
    
    st.markdown("---")
    st.markdown("**Note**: Predictions are based on trained ML model")

 
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Enter Student Details")
    
    with st.form("prediction_form"):
    
        iq = st.number_input(
            "IQ ( between 0 - 150)",
            min_value=70,
            max_value=150,
            value=100,
            step=1,
            help="Enter IQ score (typically 70-150)"
        )
        
        prev_sem_result = st.number_input(
            "Previous Semester Result",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            format="%.2f",
            help="Enter previous semester result (0-10)"
        )
        
        cgpa = st.number_input(
            "CGPA (Cumulative Grade Point Average)",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            format="%.2f",
            help="Enter CGPA (0-10)"
        )
        
        communication_skills = st.number_input(
            "Communication Skills",
            min_value=0,
            max_value=10,
            value=7,
            step=1,
            help="Enter communication skills score (0-10)"
        )
        
        projects_completed = st.number_input(
            "Projects Completed",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            help="Enter number of projects completed"
        )
        
        
        submitted = st.form_submit_button("🔮 Predict Placement", use_container_width=True)

with col2:
    st.header("📊 Prediction Results")
    
    if submitted:
        try:
            # Make prediction
            with st.spinner("Calculating prediction..."):
                percentage = st.session_state.predictor.predict(
                    IQ=iq,
                    Prev_Sem_Result=prev_sem_result,
                    CGPA=cgpa,
                    Communication_Skills=communication_skills,
                    Projects_Completed=projects_completed
                )
            
            if percentage is not None:
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                
                st.markdown(f'<div class="percentage-display">{percentage:.2f}%</div>', unsafe_allow_html=True)
                
                
                status = "Placed" if percentage >= 50 else "Not Placed"
                status_class = "placed" if percentage >= 50 else "not-placed"
                st.markdown(
                    f'<div class="status-box {status_class}">{status}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                
                st.markdown("---")
                st.subheader("📈 Interpretation")
                
                if percentage >= 80:
                    st.success("🎉 Excellent! Very high probability of placement.")
                elif percentage >= 60:
                    st.info("✅ Good chances of placement. Keep improving!")
                elif percentage >= 40:
                    st.warning("⚠️ Moderate chances. Consider improving your skills.")
                else:
                    st.error("❌ Low probability. Focus on improving academic performance and skills.")
                
                
                st.progress(percentage / 100)
                
                
                with st.expander("📋 Input Summary"):
                    st.write(f"**IQ**: {iq}")
                    st.write(f"**Previous Semester Result**: {prev_sem_result}")
                    st.write(f"**CGPA**: {cgpa}")
                    st.write(f"**Communication Skills**: {communication_skills}/10")
                    st.write(f"**Projects Completed**: {projects_completed}")
                
            else:
                st.error("❌ Error making prediction. Please check your inputs.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")
    
    else:
        st.info("👆 Fill in the form and click 'Predict Placement' to see results")

 
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Placement Prediction System | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)

