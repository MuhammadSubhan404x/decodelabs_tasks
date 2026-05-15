import time
import datetime
import streamlit as st
from aria import Chatbot

st.set_page_config(page_title="ARIA - AI Assistant", layout="wide")

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.user_name = ""
    st.session_state.bot = None

with st.sidebar:
    st.title("ARIA")
    st.caption("Rule-Based AI Assistant")
    st.divider()

    if not st.session_state.initialized:
        name_input = st.text_input("Your name", placeholder="Enter your name to start")
        if st.button("Start chat") and name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.session_state.bot = Chatbot(st.session_state.user_name)
            welcome = st.session_state.bot.welcome_message()
            st.session_state.messages.append({"role": "assistant", "content": welcome})
            st.session_state.initialized = True
            st.rerun()
    else:
        st.markdown(f"**Chatting as:** {st.session_state.user_name}")
        st.divider()
        st.markdown("**Session Stats**")
        if st.session_state.bot:
            stats = st.session_state.bot.get_session_stats()
            st.metric("Messages", stats["message_count"])
            st.metric("Duration", stats["duration"])
            if stats["topics"]:
                st.markdown("**Topics explored:**")
                for topic in stats["topics"]:
                    st.markdown(f"- {topic.replace('_', ' ').title()}")

        st.divider()
        if st.session_state.bot and st.session_state.bot.history:
            lines = []
            for entry in st.session_state.bot.history:
                speaker = st.session_state.user_name if entry["role"] == "user" else "ARIA"
                lines.append(f"{speaker}: {entry['content']}\n")
            chat_text = "\n".join(lines)
            st.download_button(
                label="Download conversation",
                data=chat_text,
                file_name=f"aria_chat_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt",
                mime="text/plain",
            )

        st.divider()
        st.markdown("**Topics I know:**")
        st.markdown("AI & Machine Learning | Python | Jokes | Motivation | About me")

st.title("ARIA Chat")
st.caption("A rule-based AI assistant with NLP-enhanced intent recognition")

if not st.session_state.initialized:
    st.info("Enter your name in the sidebar to begin.")
else:
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("ARIA is thinking..."):
                time.sleep(0.4)
                response = st.session_state.bot.respond(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
