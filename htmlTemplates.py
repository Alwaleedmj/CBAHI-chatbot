css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}

.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message"> <b>CBAHI Consultant Question: </b>
        <br> 
    {{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user"> 
    <div class="message">
     <b>Heathcare Employee Question: </b>
    <br>
    {{MSG}}</div>
</div>
'''