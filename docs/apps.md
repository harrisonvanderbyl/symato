https://twitter.com/swyx/status/1610247438958481408
ChatGPT’s current killer app isn’t search, therapy, doing math, controlling browsers, emulating a virtual machine, or any of that other cherrypicked examples that come with huge disclaimers.

It’s a lot more quotidian: __Reformatting information from any format X to any format Y.__
![](files/apps-00.jpg)

“ChatGPT reformatting” requires minimal world knowledge, are instantly verifiable, and can reliably save minutes of work multiple times a day. The reformat can include contextual inference, which saves even more time at the cost of a bit more risk: ChatGPT is capable of de-minifying JS, including adding descriptive variable names.
![](files/apps-01.png)

https://www.youtube.com/shorts/JFY4DORbyAY
Giải bài tập phổ thông trung học quốc gia
![](files/apps-02.jpg)

https://twitter.com/goodside/status/1561549768987496449
GPT-3 can translate between many disparate formats of data. For example, you can render the series premiere of Better Call Saul as a valid GraphViz dot diagram:
![](files/apps-03.jpg)

https://twitter.com/goodside
Staff Prompt Engineer @Scale_AI

https://scale.com/blog/chatgpt-vs-claude
Anthropic, an AI startup co-founded by former employees of OpenAI, has quietly begun testing a new, ChatGPT-like AI assistant named Claude. The team at Anthropic was gracious enough to grant us access, and updates to Anthropic’s social media policies mean we can now share some of our early, informal comparison findings between Claude and ChatGPT.

Overall, Claude is a serious competitor to ChatGPT, with improvements in many areas. While conceived as a demonstration of "constitutional" principles, Claude is not only more inclined to refuse inappropriate requests, but is also more fun than ChatGPT. Claude’s writing is more verbose, but also more naturalistic. Its ability to write coherently about itself, its limitations, and its goals seem to also allow it to more naturally answer questions on other subjects.

https://github.com/TheAppleTucker/backend-GPT | https://twitter.com/DYtweetshere/status/1617471632909676544
![](https://pbs.twimg.com/media/FnJpZ7VakAMTrg1?format=jpg&name=medium)
Our vision for a future tech stack is to completely replace the backend with an LLM that can both run logic and store memory. We demonstrated this with a Todo app.

We first instruct the LLM on the purpose of the backend (i.e. "This is a todo list app") and provide it with an initial JSON blob for the database state (i.e. {todo_items: [{title: "eat breakfast", completed: true}, {title: "go to school", completed: false}]}.

In every "API call" we make on the LLM,  we pass in the current state and some user-inputted instructions and extract a response for the client.
![](https://pbs.twimg.com/media/FnJpgp1aYAItoGk?format=jpg&name=medium)

A ToDo app is just one example of what's possible. We thought of LLMagnus Chess App, flash cards with LLM backends, etc. The possibilities are limitless. The future of backends might be no server, no DB, and only LLM??!

- - -

