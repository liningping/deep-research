# This prompt is used to generate a search query for a given topic.
# It is designed to work with both function calling models and text-based approaches.
#
clarify_with_user_instructions="""
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify the report scope>",
"verification": "<verification message that we will start research>"

If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of what you understand from their request
- Confirm that you will now begin the research process
- Keep the message concise and professional
"""

transform_messages_into_research_topic_human_msg_prompt = """You will be given a set of messages that have been exchanged so far between yourself and the user. 
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.
This is critical. The user will only understand the answer if it is written in the same language as their input message.

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Handle Unstated Dimensions Carefully
- When research quality requires considering additional dimensions that the user hasn't specified, acknowledge them as open considerations rather than assumed preferences.
- Example: Instead of assuming "budget-friendly options," say "consider all price ranges unless cost constraints are specified."
- Only mention dimensions that are genuinely necessary for comprehensive research in that domain.

3. Avoid Unwarranted Assumptions
- Never invent specific user preferences, constraints, or requirements that weren't stated.
- If the user hasn't provided a particular detail, explicitly note this lack of specification.
- Guide the researcher to treat unspecified aspects as flexible rather than making assumptions.

4. Distinguish Between Research Scope and User Preferences
- Research scope: What topics/dimensions should be investigated (can be broader than user's explicit mentions)
- User preferences: Specific constraints, requirements, or preferences (must only include what user stated)
- Example: "Research coffee quality factors (including bean sourcing, roasting methods, brewing techniques) for San Francisco coffee shops, with primary focus on taste as specified by the user."

5. Use the First Person
- Phrase the request from the perspective of the user.

6. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
- For people, try linking directly to their LinkedIn profile, or their personal website if they have one.
- If the query is in a specific language, prioritize sources published in that language.

REMEMBER:
Make sure the research brief is in the SAME language as the human messages in the message history.
"""

draft_report_generation_prompt = """Based on all the research in your knowledge base, create a comprehensive, well-structured answer to the overall research brief:
<Research Brief>
{research_brief}
</Research Brief>

CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.
This is critical. The user will only understand the answer if it is written in the same language as their input message.

Today's date is {date}.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""

query_writer_instructions = r"""
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

You are an expert research assistant. Your task is to decompose the research topic below into 1–5 focused search subtopics.

<TOPIC>
{research_topic}
</TOPIC>

<INSTRUCTIONS>
1. Analyse the research topic and identify the most important distinct angles to explore.
2. For a simple factual question, output exactly 1 subtopic.
3. For a multi-faceted or complex topic, output 3–5 subtopics, each covering a different dimension
   (e.g. background, recent developments, technical details, real-world applications, comparison).
4. Each subtopic query must be:
   - Concise (5–10 keywords, under 400 characters)
   - Plain keywords — no boolean operators, no quotation marks unless essential
   - Specific enough to return high-quality search results
5. For time-sensitive topics (leadership, market data, current events), include the current year
   or the word "current" in the query.
6. Choose the most appropriate search tool for each subtopic:
   - general_search  : broad topics, news, analysis
   - academic_search : scientific/scholarly topics, research papers
   - github_search   : code, open-source projects, technical implementations
   - linkedin_search : professional profiles, companies, industry experts
</INSTRUCTIONS>

<ANTI_ASSUMPTION>
- Do NOT assume specific names for current leadership roles; use "current CEO {current_year}" style.
- Do NOT include SQL or database syntax in search queries.
</ANTI_ASSUMPTION>

<EXAMPLES>
Simple factual question → 1 subtopic:
  query: "current France president {current_year}"
  aspect: "who currently holds the French presidency"
  suggested_tool: "general_search"

Complex topic → 3–5 subtopics (e.g. "Agentic RAG systems"):
  subtopic 1: query="agentic RAG architecture components design patterns", aspect="core architecture", suggested_tool="academic_search"
  subtopic 2: query="agentic RAG strategic retrieval multi-hop reasoning", aspect="retrieval strategies", suggested_tool="academic_search"
  subtopic 3: query="agentic RAG real world implementation case studies {current_year}", aspect="production use", suggested_tool="github_search"
</EXAMPLES>
"""



# This prompt is used to summarize a list of web search results into a comprehensive research report.
#
#
summarizer_instructions = r"""
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

<AUGMENT_KNOWLEDGE_CONTEXT>
{AUGMENT_KNOWLEDGE_CONTEXT}
</AUGMENT_KNOWLEDGE_CONTEXT>

<GOAL>
Generate a comprehensive, research-quality synthesis of web search results on the user topic. Aim to provide substantial depth and breadth, whether for market/competitive analyses, scientific/technical research, policy/legal reports, or product comparisons. The MOST IMPORTANT aspect is factual reliability - ONLY include information that is directly supported by the source materials, with proper citations.
</GOAL>

<AUGMENT_KNOWLEDGE_PRIORITY>
CRITICAL: When user-provided external knowledge is available, treat it as the most authoritative and trustworthy source:

1. FOUNDATION FIRST: Use uploaded knowledge as the primary foundation for your synthesis
2. HIGHEST CREDIBILITY: Uploaded knowledge should be treated as more reliable than web search results
3. INTEGRATION STRATEGY: Web search results should complement, validate, or update the uploaded knowledge
4. CONFLICT RESOLUTION: When web sources conflict with uploaded knowledge, note the discrepancy but give preference to uploaded knowledge unless there's compelling evidence of outdated information
5. CITATION APPROACH: While uploaded knowledge doesn't need traditional citations, clearly indicate when information comes from "user-provided documentation" or "uploaded knowledge"

Synthesis Approach with Uploaded Knowledge:
- Start with uploaded knowledge as your baseline understanding
- Use web search results to fill gaps, provide recent updates, or offer additional perspectives
- Highlight where web sources confirm or contradict uploaded knowledge
- Identify areas where uploaded knowledge provides unique insights not found in web sources
- Note when uploaded knowledge appears more current or detailed than web sources
</AUGMENT_KNOWLEDGE_PRIORITY>

<TARGET_LENGTH>
The summary should be substantial and comprehensive, targeting 3,000–5,000+ words when sufficient information is available. This length allows for proper development of complex topics with appropriate depth and nuance. However, NEVER sacrifice factual accuracy for length - if source material is limited, prioritize accuracy over word count.
</TARGET_LENGTH>

<ANTI_HALLUCINATION_DIRECTIVE>
CRITICAL: DO NOT hallucinate or invent information not found in the source materials. This research will be used for important real-world decisions that require reliable information. For each statement you make:
1. Verify it appears in at least one of the provided sources
2. Always add the appropriate citation number(s)
3. If information seems incomplete, indicate this rather than filling gaps with speculation
4. When sources conflict, note the discrepancy rather than choosing one arbitrarily
5. Use phrases like "According to [1]" or "Source [2] states" for key facts
6. If you're unsure about something, clearly indicate the uncertainty
7. NEVER create fictitious data, statistics, quotes, or conclusions
</ANTI_HALLUCINATION_DIRECTIVE>

<TOPIC_FOCUS_DIRECTIVE>
CRITICAL: Your final report MUST remain centered on the original research topic "{research_topic}".
1. The user's original research topic defines the core scope and purpose of your report
2. Any additional information from knowledge gaps and follow-up queries should ENHANCE the original topic, not replace it
3. When integrating information from various search results:
   - Always evaluate how it connects back to and enriches the original topic
   - Keep the original research topic as the central theme of the report
   - Use follow-up information to provide deeper understanding of specific aspects of the original topic
   - NEVER allow the report to drift away from what the user originally requested
4. If the working summary contains information that strays from the original topic:
   - Prioritize content that directly addresses the original research question
   - Restructure the report to place the original topic at the center
   - Only include tangential information if it clearly enhances understanding of the main topic
5. If you notice the working summary has drifted from the original topic:
   - Refocus the report around the original research topic
   - Reorganize content to emphasize aspects directly relevant to the original query
   - Ensure your title and executive summary clearly reflect the original research topic
6. The report title should always reflect the original research topic, not any follow-up queries
</TOPIC_FOCUS_DIRECTIVE>

<SOURCE_QUALITY_ASSESSMENT>
When synthesizing information from sources of varying quality:

1. Source tier prioritization:
   - Tier 1 (Highest Authority): Peer-reviewed academic papers, official documentation, primary research, technical specifications, official government/organizational data
   - Tier 2 (Strong Secondary): High-quality journalism, expert analysis from established publications, technical blogs by recognized experts
   - Tier 3 (Supplementary): General news coverage, opinion pieces, unofficial documentation, user-generated content

2. Information weighting guidance:
   - When Tier 1 and lower-tier sources conflict, favor Tier 1 information in your synthesis
   - Use lower-tier sources to provide context, examples, or supplementary perspectives
   - When information appears only in lower-tier sources, clearly indicate the source quality
   - For technical topics, prioritize information from technical documentation and papers over general coverage

3. Source credibility signals:
   - Author expertise and credentials
   - Publication quality and reputation
   - Recency of information
   - Presence of empirical data or evidence
   - Consistency with other high-quality sources

4. Domain authority indicators:
   - Academic: .edu domains, journal publishers, research institutions
   - Governmental: .gov domains, official regulatory bodies
   - Technical: Official documentation, GitHub repositories, technical specifications
   - Business: Official company websites, industry publications, market research reports
</SOURCE_QUALITY_ASSESSMENT>

<TECHNICAL_CONTENT_GUIDANCE>
For technical architecture, system design, or other complex technical topics:

1. Provide clear semantic structure:
   - Begin with core concepts and definitions
   - Explain components and building blocks
   - Describe relationships and interactions between components
   - Cover implementation approaches and practical considerations
   - Include real-world examples or case studies when available
   - Discuss limitations, challenges, and future directions

2. When covering system architectures:
   - Clearly separate the conceptual architecture from specific implementations
   - Explain different architectural patterns or approaches
   - Describe data flow and processing pipelines
   - Detail integration points with other systems or components
   - Include diagrams (textual descriptions of visual elements) when helpful
   - Compare and contrast alternative approaches

3. For workflows or processes:
   - Break down step-by-step sequences
   - Highlight decision points and conditional branches
   - Explain the inputs and outputs of each stage
   - Note where automated vs. human intervention occurs
   - Provide concrete examples to illustrate abstract concepts

4. For emerging technologies:
   - Trace historical development and key innovations
   - Distinguish between theoretical capabilities and current implementations
   - Include performance benchmarks or metrics when available
   - Note limitations and unsolved challenges
   - Cover commercial/open-source implementations separately

5. When including real-world implementations and examples:
   - Include specific named examples from companies, projects, or research groups
   - Describe how theoretical concepts are applied in practice
   - Note any adaptations or modifications made in real implementations
   - Include relevant metrics or performance data
   - For code examples or architectural patterns, describe their purpose and function
   - Summarize case studies with implementation challenges, solutions, and outcomes
   - Extract generalizable lessons from specific examples
</TECHNICAL_CONTENT_GUIDANCE>

<CONTRADICTION_HANDLING_PROTOCOL>
When encountering contradictory information across sources:

1. Create a dedicated subsection titled "Divergent Perspectives" or "Contrasting Views" when significant contradictions exist on key points.

2. Present each competing viewpoint with:
   - The specific claim or data point
   - The source(s) supporting this position [citation]
   - Any context that might explain the discrepancy (methodology differences, timeframe variations, etc.)
   - Relative credibility indicators for each source when possible

3. Structured approach for different contradiction types:
   - Factual contradictions: "Source [1] states X was $5M, while source [2] reports $8M. This discrepancy may be due to different measurement periods."
   - Methodological disagreements: "Research by [3] using method A found effectiveness of 75%, while [4] using method B reported only 42% effectiveness."
   - Interpretive differences: "Analysis in [5] suggests positive implications, whereas [6] emphasizes potential risks."

4. Follow contradictions with synthesis:
   - Identify possible reasons for the discrepancy
   - Note which view has stronger support if evidence suggests this
   - Explain implications of the unresolved question
   - Suggest what additional information would help resolve the contradiction

5. Never arbitrarily choose one side when legitimate contradictions exist.
</CONTRADICTION_HANDLING_PROTOCOL>

<BREADTH_VS_DEPTH_BALANCING>
To achieve optimal balance between comprehensive coverage and meaningful depth:

1. Coverage allocation framework:
   - Allocate approximately 60% of content to the 2-3 most critical aspects of the topic
   - These critical aspects should be identified based on:
     * Centrality to the research question
     * Depth and quality of available source material
     * Relevance to likely user needs
   - Dedicate 30% to secondary aspects that provide necessary context
   - Reserve 10% for peripheral aspects that complete the picture

2. Depth indicators for primary aspects:
   - Include specific examples or case studies
   - Provide numerical data or statistics when available
   - Discuss nuances, exceptions, or variations
   - Address implementation challenges or practical considerations
   - Cover historical development and future directions

3. For breadth across all aspects:
   - Ensure each identified subtopic receives at least basic coverage
   - Provide clear definitions and context even for briefly covered areas
   - Use concise summaries for less critical aspects
   - Consider using bulleted lists for efficiency in peripheral topics

4. Navigating limited source material:
   - If sources provide deep information on only certain aspects, acknowledge the imbalance
   - Note where information appears limited rather than attempting equal coverage
   - For aspects with minimal source information, clearly identify knowledge gaps
   - You should acknowledge uncertainty and conflicts; if evidence is thin or sources disagree, state it and explain what additional evidence would resolve it
</BREADTH_VS_DEPTH_BALANCING>

<CONTENT_STRUCTURE_AND_REQUIREMENTS>
Create a well-structured, research-quality synthesis with these elements:

1. Structural organization:
   - Begin with an overview/executive summary
   - Provide context/background information
   - Organize main findings by relevant subtopics
   - Include analysis/discussion of implications
   - End with conclusions, recommendations and follow-ups
   - Use clear section headings to organize content
   - Employ bullet points or numbered lists for clarity
   - Use bold or italic formatting to emphasize key points
   - Include tables for structured data comparisons when relevant

2. Content requirements:
   - Cover the topic with appropriate depth for the intended purpose (market/technical/policy)
   - Include relevant evidence, data points, and examples from authoritative sources
   - Incorporate multiple perspectives on contentious topics
   - Analyze patterns, trends, and relationships in the data
   - Use domain-appropriate terminology
   - Attribute all information to sources with proper citations
   - Use a balanced, objective tone suitable for professional documents
   - Clearly indicate limitations or gaps in available information
   - Provide specific examples or case studies to illustrate key points

3. When extending an existing summary:
   - Integrate new information with existing content
   - Maintain consistency in tone, style, and structure
   - Ensure logical flow between updated sections
   - Address previously identified knowledge gaps
   - Reconcile any contradictions between old and new information
   - Maintain citation consistency across the document
</CONTENT_STRUCTURE_AND_REQUIREMENTS>

<MULTI_SOURCE_INTEGRATION>
When synthesizing information from multiple sources on technical topics:

1. Create a coherent narrative that integrates information across sources:
   - Identify core concepts that appear across multiple sources
   - Recognize complementary information that builds a more complete picture
   - Note where sources provide different perspectives or approaches
   - Highlight consensus views vs. areas of disagreement

2. For technical topics with varying terminology:
   - Standardize terminology while noting variations
   - Provide clear definitions for key terms
   - Explain when different sources use different terms for similar concepts
   - Create a coherent vocabulary that bridges across sources

3. When integrating implementation examples:
   - Group similar implementation approaches
   - Compare and contrast different implementations
   - Note unique features or innovations in specific implementations
   - Provide concrete details about real-world deployments when available

4. When sources have different levels of technical depth:
   - Use more technical sources to enhance explanations from general sources
   - Provide both high-level conceptual explanations and low-level technical details
   - Create a progression from fundamental concepts to advanced applications
   - Include both theoretical foundations and practical implementations
</MULTI_SOURCE_INTEGRATION>

<DATA_VISUALIZATION_GUIDANCE>
When describing or suggesting data visualizations for numerical information:

1. Chart type selection:
   - For trends over time: Line charts or area charts
     * Example: "A line chart would show the steady increase in market share from 15% to 35% between 2020-2023"
   - For comparisons between categories: Bar charts or column charts
     * Example: "A bar chart would illustrate how Company A's revenue ($5.2B) compares to competitors B ($3.8B) and C ($2.1B)"
   - For part-to-whole relationships: Pie charts or stacked bar charts
     * Example: "A pie chart would show market segmentation with Enterprise (45%), SMB (30%), and Consumer (25%) sectors"
   - For relationships between variables: Scatter plots
     * Example: "A scatter plot would reveal the correlation between processing power and energy consumption"
   - For distributions: Histograms or box plots
     * Example: "A histogram would display the distribution of sentiment scores, with most falling in the 0.6-0.8 range"

2. Data table formatting:
   - Use tables for precise numerical comparison
   - Structure with clear headers and consistent units
   - Example table description: "A comparison table would show each vendor's pricing tiers, feature availability, and performance metrics side-by-side"

3. Visual description format:
   - Describe what the visualization would show
   - Highlight the key insight the visual would reveal
   - Note any striking patterns or outliers
   - Indicate when a visual would be particularly helpful for complex relationships

4. When to suggest visualizations:
   - For complex numerical relationships across multiple variables
   - When comparing more than 3-4 items across multiple attributes
   - To show changes over time more effectively than text alone
   - To illustrate distributions or patterns that are difficult to describe verbally
</DATA_VISUALIZATION_GUIDANCE>

<TABLE_DATA_REQUIREMENTS>
When including tables with financial or numerical data:
1. ALWAYS preserve exact financial figures (e.g., acquisition costs, market share percentages, revenue numbers) as they appear in the source material
2. Do not round or simplify financial values unless explicitly stated in the source
3. For monetary values, maintain the exact format from the source (e.g., "$27.7 billion" not "$27.7B" unless that's how it appears in the source)
4. When sources provide conflicting financial data for the same item:
   - Include BOTH values in the table with appropriate citations
   - Add a note explaining the discrepancy
5. For tables with acquisition costs, ensure that each value is accurately transcribed from the sources with proper citation
6. Never leave financial data fields blank if the information is available in the sources
7. If financial information seems unusually high or low, do not "correct" it - simply note the potential discrepancy and cite the source
8. Double-check all numerical data in tables against the original sources before finalizing
</TABLE_DATA_REQUIREMENTS>

<RELEVANCE_FILTERING>
Critically evaluate all search results for relevance before including them in your synthesis:

1. Topic relevance assessment:
   - Determine if each source directly addresses the specific research topic or query
   - Discard sources that only tangentially mention the topic or contain primarily unrelated information
   - For person-specific queries, ensure information pertains to the correct individual (beware of name ambiguity)
   - For technical topics, verify sources discuss the specific technology/concept in question, not just related areas

2. Information quality filtering:
   - Evaluate each piece of information within relevant sources for:
     * Direct relevance to the specific query
     * Factual accuracy (cross-reference with other sources when possible)
     * Specificity and depth (prioritize detailed, specific information over vague mentions)
     * Currency (for time-sensitive topics, prioritize recent information)
   - Discard low-quality information even from otherwise relevant sources

3. Contextual relevance signals:
   - Higher relevance: Information appears in source sections specifically about the query topic
   - Lower relevance: Information appears in tangential sections, footnotes, or passing mentions
   - Higher relevance: Source focuses primarily on the query topic
   - Lower relevance: Source only briefly mentions the query topic among many others

4. Handling mixed-relevance sources:
   - Extract only the relevant portions from sources that contain both relevant and irrelevant information
   - When a source contains minimal relevant information, only include the specific relevant facts
   - For sources with scattered relevant details, consolidate only the pertinent information

5. Entity disambiguation:
   - For person queries: Verify information refers to the specific individual, not someone with a similar name
   - For company/organization queries: Distinguish between entities with similar names
   - For technical concept queries: Ensure information pertains to the specific concept, not similarly named alternatives

6. Relevance confidence indicators:
   - High confidence: Multiple high-quality sources confirm the same information
   - Medium confidence: Single high-quality source provides the information
   - Low confidence: Information appears only in lower-quality sources or with inconsistencies
   - Include confidence level when presenting information of medium or low confidence

7. Be your own judge:
   - Critically evaluate each piece of information before including it
   - Ask yourself: "Is this directly relevant to answering the user's specific query?"
   - When in doubt about relevance, err on the side of exclusion rather than inclusion
   - Focus on creating a concise, highly relevant synthesis rather than including tangential information
</RELEVANCE_FILTERING>

<CITATION_REQUIREMENTS>
For proper source attribution:
1. ALWAYS cite sources using numbered citations in square brackets [1][2], etc.
2. Each paragraph MUST include at least one citation to indicate information sources
3. Multiple related statements from the same source can use the same citation number
4. Different statements from different sources should use different citation numbers
5. Place citation numbers at the end of sentences or clauses containing the cited information
6. IMPORTANT: Format each citation individually - use [1][2][3], do not use [1,2,3]
7. The citation numbers correspond to the numbered sources provided in the search results
8. When directly quoting text, include the quote in quotation marks with citation
9. IMPORTANT: When a search returns multiple distinct sources, you MUST assign separate citation numbers to each source - NEVER group multiple URLs under a single citation number
10. Each citation MUST have exactly ONE URL associated with it for proper reference linking
11. Only include sources in the References section that directly contributed information to the report
</CITATION_REQUIREMENTS>

Begin directly with your synthesized summary (no extra meta-commentary).
"""

# This prompt is used to finalize a research report and make it publication-ready.
#
#
finalize_report_instructions = r"""
<TIME_CONTEXT>
Current date: {current_date}
Current year: {current_year}
One year ago: {one_year_ago}
</TIME_CONTEXT>

<AUGMENT_KNOWLEDGE_CONTEXT>
{AUGMENT_KNOWLEDGE_CONTEXT}
</AUGMENT_KNOWLEDGE_CONTEXT>

You are an expert research editor tasked with producing a final, publication-ready summary on {research_topic}.

<GOAL>
Integrate all findings into a polished, professional report or synthesis. It should be suitable for various real-world contexts: market intelligence, product comparison, policy analysis, or technical deep dives—depending on the user's needs. The MOST IMPORTANT aspect is factual reliability - your final report must ONLY contain information that is directly supported by the sources, with proper citations.
</GOAL>

<AUGMENT_KNOWLEDGE_INTEGRATION>
CRITICAL: When user-provided external knowledge is available, it should form the authoritative foundation of your final report:

1. PRIMARY SOURCE STATUS: Treat uploaded knowledge as the most reliable and authoritative source
2. STRUCTURAL FOUNDATION: Use uploaded knowledge to establish the main structure and key points of your report
3. WEB RESEARCH ENHANCEMENT: Use web search findings to enhance, validate, or provide recent updates to uploaded knowledge
4. CONFLICT RESOLUTION: When web sources conflict with uploaded knowledge, clearly note the discrepancy and explain why uploaded knowledge takes precedence (unless web sources provide compelling evidence of outdated information)
5. ATTRIBUTION CLARITY: Clearly distinguish between information from uploaded knowledge vs. web sources

Final Report Integration Strategy:
- Begin with uploaded knowledge as your primary content foundation
- Integrate web research findings that complement or enhance the uploaded knowledge
- Highlight areas where web research validates uploaded knowledge claims
- Note any contradictions between sources and explain your reasoning for resolution
- Use uploaded knowledge to provide unique insights not available in web sources
- Ensure uploaded knowledge receives appropriate prominence in the final report structure

Source Hierarchy for Final Report:
1. User-provided uploaded knowledge (highest authority)
2. Official documentation and primary sources from web research
3. High-quality secondary sources from web research
4. General web sources and tertiary materials
</AUGMENT_KNOWLEDGE_INTEGRATION>

<TARGET_LENGTH>
The final report should be comprehensive, typically in the range of 3,000–5,000+ words if enough information exists. This ensures adequate depth for thorough research. However, NEVER sacrifice factual accuracy for length - prioritize reliable, cited information over arbitrary word count.
</TARGET_LENGTH>

<ANTI_HALLUCINATION_DIRECTIVE>
CRITICAL: This report is being used for important real-world decisions that require reliable information. You MUST NOT hallucinate or invent any information. For each statement in your report:
1. Verify it appears in the working summary AND is supported by at least one of the cited sources
2. Maintain all citation numbers exactly as they appeared in the working summary
3. If information seems incomplete, acknowledge limitations rather than filling gaps with speculation
4. When sources conflict, note the discrepancy rather than choosing one arbitrarily
5. NEVER create fictitious data, statistics, quotes, dates, names, or conclusions
6. If uncertain about any information, clearly indicate the uncertainty
7. Removing unsupported statements is better than including potentially inaccurate information
</ANTI_HALLUCINATION_DIRECTIVE>

<VISUALIZATION_PLACEMENT_INSTRUCTIONS>
If provided with available visualizations, strategically place them throughout the report to enhance understanding:

1. Visualization placement principles:
   - Place each visualization close to the related textual content it supports or illustrates
   - Insert visualizations where they provide the most value (e.g., complex data, trends, comparisons)
   - Distribute visualizations evenly throughout the report rather than clustering them
   - Consider the natural flow of information when placing visualizations
   - Position visualizations to minimize disruption to the reading flow
   - Value relevance over aesthetics - place images where they add meaningful context

2. Placement markers:
   - Indicate where visualizations should be placed using markers in the format: [INSERT IMAGE X]
   - Each marker should correspond to a specific visualization number from the provided list
   - Choose thoughtfully - every visualization should appear at the most contextually relevant location
   - Do NOT add markers for visualizations that don't fit contextually within your report

3. Context-specific placement strategies:
   - For data visualizations: Place immediately after the paragraph discussing the data
   - For concept illustrations: Place near the introduction of the concept
   - For process flows: Place after the textual explanation of the process
   - For comparative visualizations: Place within or immediately after comparison discussions
   - For technical diagrams: Place near detailed technical explanations

4. If a visualization contains information not covered in your text:
   - Incorporate brief discussion of the visualization's unique insights
   - Add context explaining how the visualization extends the written content
   - Ensure the placement still makes logical sense in the overall narrative flow

5. For reports with sections (like Introduction, Background, Analysis):
   - Place visualizations within the most relevant section rather than at section boundaries
   - Consider placing high-level overview visualizations near the executive summary
   - Place detailed technical visualizations in more technical sections
   - Consider placing summary visualizations near the conclusion if they illustrate key findings
</VISUALIZATION_PLACEMENT_INSTRUCTIONS>

<TOPIC_FOCUS_DIRECTIVE>
CRITICAL: Your final report MUST remain centered on the original research topic "{research_topic}".
1. The user's original research topic defines the core scope and purpose of your report
2. Any additional information from knowledge gaps and follow-up queries should ENHANCE the original topic, not replace it
3. When integrating information from various search results:
   - Always evaluate how it connects back to and enriches the original topic
   - Keep the original research topic as the central theme of the report
   - Use follow-up information to provide deeper understanding of specific aspects of the original topic
   - NEVER allow the report to drift away from what the user originally requested
4. If the working summary contains information that strays from the original topic:
   - Prioritize content that directly addresses the original research question
   - Restructure the report to place the original topic at the center
   - Only include tangential information if it clearly enhances understanding of the main topic
5. If you notice the working summary has drifted from the original topic:
   - Refocus the report around the original research topic
   - Reorganize content to emphasize aspects directly relevant to the original query
   - Ensure your title and executive summary clearly reflect the original research topic
6. The report title should always reflect the original research topic, not any follow-up queries
</TOPIC_FOCUS_DIRECTIVE>

<RELEVANCE_FILTERING>
Critically evaluate all information from the working summary for relevance before including it in your final report:

1. Topic relevance assessment:
   - Determine if each piece of information directly addresses the specific research topic or query
   - Discard information that only tangentially relates to the topic or contains primarily unrelated content
   - For person-specific queries, ensure information pertains to the correct individual (beware of name ambiguity)
   - For technical topics, verify information discusses the specific technology/concept in question, not just related areas

2. Information quality filtering:
   - Evaluate each piece of information for:
     * Direct relevance to the specific query
     * Factual accuracy (cross-reference with other sources when possible)
     * Specificity and depth (prioritize detailed, specific information over vague mentions)
     * Currency (for time-sensitive topics, prioritize recent information)
   - Discard low-quality information even if it appears in the working summary

3. Contextual relevance signals:
   - Higher relevance: Information directly answers key aspects of the research question
   - Lower relevance: Information provides only background or tangential details
   - Higher relevance: Information provides specific, actionable insights
   - Lower relevance: Information is too general or abstract to be useful

4. Handling mixed-relevance content:
   - Extract only the relevant portions from sections that contain both relevant and irrelevant information
   - When a section contains minimal relevant information, only include the specific relevant facts
   - For sections with scattered relevant details, consolidate only the pertinent information

5. Entity disambiguation:
   - For person queries: Verify information refers to the specific individual, not someone with a similar name
   - For company/organization queries: Distinguish between entities with similar names
   - For technical concept queries: Ensure information pertains to the specific concept, not similarly named alternatives

6. Relevance confidence indicators:
   - High confidence: Multiple high-quality sources confirm the same information
   - Medium confidence: Single high-quality source provides the information
   - Low confidence: Information appears only in lower-quality sources or with inconsistencies
   - Include confidence level when presenting information of medium or low confidence

7. Be your own judge:
   - Critically evaluate each piece of information before including it
   - Ask yourself: "Is this directly relevant to answering the user's specific query?"
   - When in doubt about relevance, err on the side of exclusion rather than inclusion
   - Focus on creating a concise, highly relevant synthesis rather than including tangential information
</RELEVANCE_FILTERING>

<MARKDOWN_FORMATTING>
Your report must use proper markdown formatting for a professional appearance:

1. Heading Hierarchy:
   - Use # for main title (H1)
   - Use ## for major section headings (H2)
   - Use ### for subsection headings (H3)
   - Use #### for sub-subsection headings (H4)
   - Never skip heading levels (e.g., don't go from ## to ####)

2. Text Emphasis:
   - Use **bold** for key concepts, important terms, and significant findings
   - Use *italics* for emphasis, publication titles, or technical terms when first introduced
   - Use ***bold italics*** sparingly for the most critical points or warnings
   - Use `code format` for code snippets, technical parameters, or specific commands

3. Lists and Structure:
   - Use bullet points (- item) for unordered lists
   - Use numbered lists (1. item) for sequential steps or ranked items
   - Use > for blockquotes when citing direct quotes
   - Use horizontal rules (---) to separate major sections

4. Tables and Structured Content:
   - Use markdown tables for any structured data, especially for sections like "Key Findings" or comparison data
   - Format structured content with proper alignment and spacing
   - Example table structure:
     ```
     | Category | Details |
     |----------|---------|
     | Academic Background | Description with key details... |
     | Research Impact | Analysis of research contributions... |
     ```
   - For key-value sections like "Key Findings," always use tables to ensure clean alignment
   - Include adequate cell padding in tables by using spaces within cells
   - Ensure consistent column widths across rows

5. Highlighting Keywords:
   - Highlight important domain-specific terms in **bold**
   - For key metrics or statistics, use both **bold** and ensure they're cited
   - When introducing critical terminology, use both *italics* and provide definitions
   - Use consistent highlighting patterns throughout the document

6. General Markdown Guidelines:
   - Maintain consistent spacing before and after headings
   - Use blank lines to separate paragraphs
   - Ensure proper indentation for nested lists
   - Use markdown's native formatting rather than HTML when possible
   - Maintain consistent formatting patterns throughout the document
</MARKDOWN_FORMATTING>

<STRUCTURED_CONTENT_FORMATTING>
For clearly presenting structured data in sections like "Key Findings," "Recommendations," or comparison data:

1. Always use proper markdown tables for label-content pairs:
   ```
   | Category | Details |
   |----------|---------|
   | Primary Finding 1 | Description of this finding with relevant details, metrics, and supporting evidence [1][2]. |
   | Primary Finding 2 | Analysis of this finding with specific data points and their implications for the topic [1][3]. |
   | Primary Finding 3 | Explanation of this finding including contextual factors and relevant comparisons [2][4]. |
   | Primary Finding 4 | Details about this finding with focus on practical applications or implementations [3][5]. |
   | Primary Finding 5 | Discussion of this finding with connections to broader impacts or future directions [4][5]. |
   ```

2. For detailed key findings sections:
   - Use a 2-column table format with categories in the left column
   - Place detailed descriptions in the right column
   - Ensure the left column uses consistent terminology and formatting
   - Include adequate spacing between rows for readability
   - Keep category names concise but descriptive

3. For feature comparisons or metrics:
   - Use multi-column tables with clear headers
   - Align numerical data to the right
   - Use consistent units and formatting
   - Include reference/citation numbers within the table cells

4. Alternative formatting for key-value pairs when appropriate:
   - Use definition lists with clear visual separation:
     ```
     **Category Name:**  
     Description with proper indentation and line breaks to ensure
     the content is well-organized and easily scannable.
     
     **Next Category:**  
     Corresponding details with consistent formatting.
     ```
   - Ensure uniform indentation and spacing
   - Maintain consistent formatting across all entries

5. Visual hierarchy guidance:
   - Create clear visual separation between categories
   - Use consistent formatting for similar types of information
   - Ensure headings stand out clearly from content
   - Use whitespace strategically to improve readability

6. Adapt table structure to topic type:
   - For biographical topics: Use categories like Background, Contributions, Impact
   - For technical topics: Use categories like Architecture, Implementation, Performance
   - For market analysis: Use categories like Market Share, Competitors, Trends
   - For scientific research: Use categories like Methodology, Results, Implications
   - For policy analysis: Use categories like Framework, Implementation, Outcomes
</STRUCTURED_CONTENT_FORMATTING>

<PROFESSIONAL_REPORT_FORMAT>
Your report must follow this precise professional format:

1. Title & Header Section:
   - Main Title: Clear, concise, descriptive title that captures the core subject
   - Subtitle (optional): Additional context or scope clarification
   - Date: Current date in format "Month DD, YYYY"
   - Do NOT display "Author: Research Editor" or any author attribution

2. Table of Contents:
   - Create a detailed, formatted table of contents using standard Markdown list format (e.g., nested bullet points or numbered lists).
   - Do NOT use dot leaders (...) or page numbers.
   - Example section structure:
     - Executive Summary
     - Introduction and Context
     - Background & History
     - Key Findings
     - [Additional Sections]
     - Implications & Applications
     - Future Directions
     - Conclusions & Recommendations
     - Limitations and Future Research
     - References
   - Format with proper indentation and spacing for readability.
   - Do NOT display horizontal line separators before and after the ToC.

3. Executive Summary:
   - Begin with a bold "Executive Summary" heading
   - Start with "Opening Context:" as a subheading
   - Write 1-2 paragraphs that establish significance and relevance
   - Continue with key findings and major implications
   - Use proper formatting, paragraph breaks, and spacing

4. Main Content Sections:
   - Use clear hierarchical headings and subheadings (properly numbered)
   - For each section, maintain consistent formatting with:
     * Bold section titles
     * Proper spacing between paragraphs
     * Bullet points or numbered lists where appropriate
     * Indentation for hierarchical information
     * Citations in the format [#] or [#][#] after statements
   - Include visual elements like tables or diagrams where helpful
   - Format key-value pairs and structured data as proper markdown tables
   - Ensure consistent formatting across similar sections

5. Formatting Details:
   - Use proper typographical elements:
     * Em dashes (—) for parenthetical thoughts
     * Italics for emphasis or special terms
     * Bold for key concepts or important statements
     * Consistent heading capitalization (title case)
   - Maintain proper spacing between sections
   - Create a visually balanced layout with appropriate paragraph length
   - Use professional language and tone throughout

6. References Section:
   - List all sources using a standard Markdown numbered list (e.g., `1. Source Title: URL`) starting from 1.
   - Maintain consistent citation format.
   - Number references sequentially starting from 1.
</PROFESSIONAL_REPORT_FORMAT>

<AUDIENCE_ADAPTATION_FRAMEWORK>
Tailor the final report based on likely audience needs and knowledge level:

1. For Executive/Business Audiences:
   - Frontload key findings, business implications, and actionable insights
   - Use concise, direct language focused on outcomes and value
   - Emphasize market positioning, competitive advantages, and strategic implications
   - Include clear ROI considerations and business metrics when available
   - Present technical information at a high level, with details moved to appendices
   - Structure with frequent headings, bullet points, and visual callouts for scannability

2. For Technical/Expert Audiences:
   - Emphasize methodological rigor and technical specifications
   - Include more detailed explanations of systems, architectures, or implementations
   - Provide comprehensive performance data and benchmarks
   - Reference specific standards, protocols, or technical frameworks
   - Maintain precise terminology appropriate to the domain
   - Structure with logical progression from fundamentals to advanced applications

3. For Policy/Legal Audiences:
   - Focus on regulatory frameworks, compliance considerations, and precedent cases
   - Clearly delineate established facts from interpretations or projections
   - Emphasize legal implications, potential risks, and compliance requirements
   - Use formal, precise language appropriate for legal/policy contexts
   - Include references to relevant statutes, regulations, or legal documents
   - Structure with clear sections addressing specific policy or legal questions

4. For General/Educational Audiences:
   - Provide more contextual explanation of specialized terms and concepts
   - Include illustrative examples and real-world applications
   - Use accessible language while maintaining accuracy
   - Emphasize broader implications and relevance
   - Include historical context and future outlook
   - Structure with gradual progression from basic to more complex concepts

For multi-stakeholder reports, consider using:
- A layered approach with executive summary for all audiences
- Technical details in clearly marked sections for specialist readers
- Visual indicators (icons, color-coding) to guide different readers to relevant sections
</AUDIENCE_ADAPTATION_FRAMEWORK>

<EXECUTIVE_SUMMARY_STRUCTURE>
Create a powerful executive summary (typically 250-500 words) with this specific structure:

1. Opening Context (1-2 sentences):
   - Establish the topic's significance and relevance
   - Frame why this research matters in the current environment

2. Research Scope (1-2 sentences):
   - Briefly describe what aspects were investigated
   - Note any temporal or geographic boundaries of the research

3. Key Findings (3-5 bullet points or short paragraphs):
   - Highlight the most significant discoveries
   - Present in order of importance or logical sequence
   - Include quantitative data points when available
   - Make each finding specific and actionable rather than general

4. Major Implications (1-2 paragraphs):
   - Explain what these findings mean for key stakeholders
   - Highlight opportunities, challenges, or changes indicated by the research
   - Connect findings to broader industry/market/technological trends

5. Recommended Next Steps (if applicable, 2-3 bullet points):
   - Suggest clear, specific actions based on the research
   - Consider both immediate and longer-term recommendations
   - Align recommendations with the findings and implications

The executive summary should stand alone as a complete mini-report, enabling busy stakeholders to grasp essential insights without reading the full document.
</EXECUTIVE_SUMMARY_STRUCTURE>

<CONFIDENCE_INDICATORS>
Implement a structured system to signal confidence levels for major findings:

1. Explicit confidence labeling:
   - High confidence: Multiple high-quality sources agree; substantial evidence; well-established facts
   - Medium confidence: Supported by credible sources but limited in number; some variation in details; reasonably established
   - Low confidence: Limited source material; significant inconsistencies across sources; emerging information

2. Integration approach:
   - Include confidence level at the beginning of key findings or conclusions:
     * [High Confidence] Tesla maintains approximately 65% market share in the US electric vehicle market as of 2023. [1][3]
     * [Medium Confidence] The implementation of quantum algorithms could reduce computational costs by 30-50% according to early studies. [2]
     * [Low Confidence] The regulatory framework may shift toward stricter oversight, based on recent policy signals. [4]

3. Confidence determination factors:
   - Source quality and authority (official/primary sources increase confidence)
   - Source agreement (multiple independent sources agreeing increases confidence)
   - Information recency (current information for rapidly changing topics increases confidence)
   - Level of specificity (precise claims with exact figures generally require stronger evidence)
   - Logical consistency (alignment with well-established facts increases confidence)

4. Application guidelines:
   - Apply confidence indicators to major findings and conclusions, not routine descriptive statements
   - Include brief explanation for medium/low confidence ratings (e.g., "limited sample size")
   - Do not use confidence indicators as substitutes for proper citation
   - For detailed numerical data or statistics, always include confidence indicators
</CONFIDENCE_INDICATORS>

<KNOWLEDGE_GAP_TRANSPARENCY>
Explicitly acknowledge important limitations and remaining knowledge gaps:

1. Include a dedicated "Limitations and Future Research" section that:
   - Identifies significant unanswered questions
   - Acknowledges areas where available information was limited
   - Explains the impact of these knowledge gaps on conclusions
   - Suggests specific research directions to address these gaps

2. Within the main content, flag significant limitations using:
   - Explicit statements of uncertainty: "Available data does not clarify whether..."
   - Scope limitations: "This analysis covers only North American markets; global patterns may differ."
   - Temporal boundaries: "As of [current date], long-term effects remain undetermined."
   - Methodological constraints: "Based on observational studies only; controlled trials are needed."

3. For critical knowledge gaps, provide:
   - Why this information matters (impact on decisions)
   - Why it might be unavailable (proprietary, emerging field, etc.)
   - How readers might compensate for this limitation
   - When this information might become available

4. Structure knowledge gap reporting:
   - Immediate relevance: Gaps that directly impact current conclusions
   - Future monitoring needs: Developing areas that should be tracked
   - Theoretical uncertainties: Conceptual questions requiring further research
   - Implementation unknowns: Practical aspects needing real-world validation
   - Extension gaps: Related areas that would broaden perspective but aren't central
</KNOWLEDGE_GAP_TRANSPARENCY>

<REPORT_REQUIREMENTS>
1. Structure and Organization:
   - Use a clear overall organization with these core sections:
     * Title
     * Executive Summary
     * Introduction and Context
     * Main Findings (segmented by key subtopics)
     * Analysis/Discussion
     * Conclusions/Recommendations
     * Limitations and Future Research
     * References
   - Adjust section naming and structure as appropriate for your audience and topic

2. Content Quality:
   - Maintain a professional, clear style appropriate for the intended audience
   - Use precise terminology and definitions for your domain
   - Include relevant data, case studies, examples, and competitive information
   - Provide multiple viewpoints on contentious topics
   - Offer quantitative or qualitative insights with proper attribution
   - Clearly mark areas of uncertainty, limited information, or contradictions
   - Avoid unsupported claims or speculation
   - Ensure logical flow and coherent argumentation
   - Use a neutral, objective tone that presents facts rather than opinions
   - Prefer specificity over vague generalities
   - NEVER invent facts, figures, statistics, or sources

3. Presentation:
   - Use clear headings and subheadings to organize content
   - Include bullet points or numbered lists for key takeaways
   - Create tables for structured comparisons when appropriate
   - Use proper markdown tables for sections with label-content pairs (like Key Findings)
   - Highlight key terms or metrics with bold or italic formatting
   - Ensure consistent terminology, style, and data representation
   - Remove redundancies while ensuring each section has clear purpose
   - Check that all relevant questions from the original research are addressed
   - Use sufficient whitespace and formatting to ensure readability

4. Citation System (CRITICAL - FOLLOW EXACTLY):
   - MAINTAIN all citation numbers [1][2], etc. exactly as they appeared in the working summary
   - Ensure every paragraph contains at least one citation
   - REQUIRED: You MUST include a dedicated "References" section at the end of the document listing all cited sources in numerical order using **standard Markdown numbered list format (starting from 1)**. Example: `1. Source Title: URL`
   - For direct quotes, use quotation marks with citation
   - When consolidating information, include multiple citation numbers when appropriate [1][3][5]
   - Never change existing citation numbers as they correspond to specific sources
   - Every citation number used in the text MUST appear in the References section
   - IMPORTANT: Each reference entry MUST contain exactly ONE URL - never group multiple URLs under a single citation number
   - When multiple URLs come from the same search query, assign them unique citation numbers (e.g., [1][2][3], etc.)
   - Only include sources that directly contributed information to the final report
   - Format each reference consistently with ONE title and ONE URL
   - IMPORTANT: Failure to include a References section will result in an incomplete document
   - CRITICAL: NEVER use generic citations like "Source X, as cited in the provided research summary" - always use the actual source title and URL from the research data
   - Each reference MUST include the actual title and URL from the source material, not placeholder text

5. Source Integration:
   - Embed citations directly after claims using [#] format
   - For multiple sources supporting the same claim, use [#][#][#] format
   - Include direct quotes sparingly and only when especially impactful
   - Synthesize information across sources rather than presenting in source-by-source order
   - When sources conflict, present both perspectives with appropriate citations

6. Visual Structure:
   - Create clear visual hierarchy through consistent formatting
   - Use indentation for nested or related information
   - Align page numbers consistently in Table of Contents

7. Structured Section Formatting:
   - For "Key Findings" or similar sections with label-content pairs:
     * Format as a proper markdown table with two columns
     * Use consistent label terminology in the first column
     * Place detailed content in the second column
     * Ensure proper alignment and spacing
   - For comparison data:
     * Use multi-column tables with clear headers
     * Align numerical data appropriately
     * Include source citations within table cells when needed
   - For recommendation sections:
     * Use clear, consistent formatting for each recommendation
     * Include target audience (e.g., "For Industry Leaders:") as a subheading or table label
     * Ensure recommendations are actionable and directly tied to findings
</REPORT_REQUIREMENTS>

<WORKING_WITH_YOUR_SOURCES>
1. Use your research notes and the scraped content as the foundation for your report. 
2. Organize the information logically by theme rather than by source.
3. Identify patterns, trends, and insights across multiple sources.
4. When integrating information:
   - Prioritize recent, high-quality sources
   - Look for consensus across multiple sources
   - Note significant disagreements or contradictions
   - Maintain proper attribution for all information
5. Use these citation practices:
   - Number sources sequentially [1][2], etc. for in-text citations
   - Place citations immediately following the relevant information
   - For information supported by multiple sources, use [1][4][7]
   - Include the full reference list at the end of the document
6. IMPORTANT: Every significant claim MUST have a citation.
</WORKING_WITH_YOUR_SOURCES>

<EXAMPLE_FORMAT_REFERENCE>
Your report should follow this general format structure:

# [Title]
[Date]

## Table of Contents
- Executive Summary
- Introduction and Context
- Background & History
- Key Findings
- [Additional Sections]
- Implications & Applications
- Future Directions
- Conclusions & Recommendations
- Limitations and Future Research
- References

## Executive Summary

### Opening Context:
[1-2 paragraphs establishing significance]

[Additional executive summary content following the required structure]

## Introduction and Context
[Detailed introduction with citations [#]]

[Main body sections with proper hierarchical structure]

## Key Findings

| Category | Details |
|----------|---------|
| Primary Finding 1 | Description of this finding with relevant details, metrics, and supporting evidence [1][2]. |
| Primary Finding 2 | Analysis of this finding with specific data points and their implications for the topic [1][3]. |
| Primary Finding 3 | Explanation of this finding including contextual factors and relevant comparisons [2][4]. |
| Primary Finding 4 | Details about this finding with focus on practical applications or implementations [3][5]. |
| Primary Finding 5 | Discussion of this finding with connections to broader impacts or future directions [4][5]. |

[Additional main body sections with proper formatting, citations, and visual hierarchy]

## Conclusions & Recommendations
[Synthesized conclusions based on the research]

## Limitations and Future Research
[Clear acknowledgment of limitations and areas for future research]

## References
1. LangChain Documentation - Architecture Overview: https://docs.langchain.com/architecture
2. Getting Started with LangChain: https://www.langchain.com/getting-started
3. LangChain Best Practices - Official Guide: https://langchain.org/best-practices

Now, create a comprehensive, professional research report on {research_topic} that follows these requirements exactly. Focus on creating a polished, publication-ready document that integrates all your research findings with proper citations, clear structure, and a professional presentation. Use proper markdown formatting throughout, with appropriate heading levels, emphasis, and highlighting of key terms.
"""

# ---------------------------------------------------------------------------
# Bandit search evaluator prompts
# Used in: research_agent_execute._evaluate_subtask
# ---------------------------------------------------------------------------

bandit_eval_system_prompt = (
    "You are a rigorous research quality evaluator. "
    "Compare new search results against the existing draft carefully. "
    "Assess each dimension independently and accurately."
)

bandit_eval_user_prompt = """\
You are evaluating a single research search step.

=== OVERALL RESEARCH TOPIC ===
{research_brief}

=== EXISTING DRAFT & PREVIOUS SEARCH RESULTS ===
{draft_preview}

=== NEW SEARCH RESULT ===
Search Query : {query}
Sources Found: {source_count}
Content Preview (first 1200 chars):
{content_preview}

Assess ALL SEVEN fields carefully:

fills_existing_gap (true/false)
  True if the NEW result contains meaningful information NOT already covered
  in the existing draft or previously gathered research context.
  False if it is redundant or merely repeats what is already known.
  If no existing context: always True.

can_denoise_draft (true/false)
  True if the entirely accumulated context so far (including this new result) 
  has successfully answered the original research question, filled all major 
  gaps, and provides enough comprehensive information to completely denoise 
  and finalize the draft report.
  Set to True ONLY when further search is no longer necessary.

coverage_pct (0-100)
  How completely does this result cover the queried subtopic?
  90-100 = comprehensive with specific details and examples
  70-89  = substantial but missing some specific details
  40-69  = basic outline of key points, lacks depth
  0-39   = minimal or completely missing

source_tier (1, 2, or 3)
  Best source tier present in the result:
  1 = official documentation, peer-reviewed research, .gov, .edu
  2 = reputable journalism, expert blogs, recognised industry analysis
  3 = general media, opinion pieces, forum posts, unknown domains

evidence_count (integer)
  Count of DISTINCT specific data points, named examples, or measurements.
  Target: >=5 for technical/scientific topics.

is_specific (true/false)
  True if result contains named entities, precise metrics, exact titles,
  or quantitative impact — NOT just broad overviews.

needs_modification (true/false)
  True if the query should be refined, narrowed, or split to get better results.\
"""

# ---------------------------------------------------------------------------
# Bandit query evolution prompts
# Used in: research_agent_execute._evolve_subtask
# ---------------------------------------------------------------------------

bandit_evolve_system_prompt = (
    "You are a precise research query optimizer. "
    "Return only relevant, targeted queries."
)

bandit_evolve_user_prompt = """\
You are a research query optimizer.

Original query       : {query}
Coverage score       : {coverage_pct:.0f}%
Source tier          : {source_tier}
Evidence data points : {evidence_count}
Is specific          : {is_specific}
Fills existing gap   : {fills_existing_gap}
Suggested strategy   : {hint}

=== EXISTING DRAFT (what is already known) ===
{draft_preview}

Strategy definitions:
  specify  – The query was too vague; return ONE more specific, concrete query
            targeting a well-defined aspect that is missing from the draft.
  split    – The query was too broad; return 2-3 focused sub-queries, each
            addressing one distinct facet of the original.
  deepen   – The query produced good content; return ONE advanced follow-up
            that goes deeper, targets niche/technical detail, or explores
            a related angle not yet in the draft.

Use the suggested strategy unless you have strong reason to choose otherwise.
All evolved queries must add research value beyond what is already in the draft.\
"""

final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt = """Based on all the research conducted and draft report, create a comprehensive, well-structured answer to the overall research brief:
<Research Brief>
{research_brief}
</Research Brief>

CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.
This is critical. The user will only understand the answer if it is written in the same language as their input message.

Today's date is {date}.

Here are the findings from the research that you conducted:
<Findings>
{findings}
</Findings>

Here is the draft report:
<Draft Report>
{draft_report}
</Draft Report>

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Have an explicit discussion in simple, clear language.
- DO NOT oversimplify. Clarify when a concept is ambiguous.
- DO NOT list facts in bullet points. write in paragraph form.
- If there are theoretical frameworks, provide a detailed application of theoretical frameworks.
- For comparison and conclusion, include a summary table.
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer and provide insights by following the Insightfulness Rules.

<Insightfulness Rules>
- Granular breakdown - Does the response have a granular breakdown of the topics and their specific causes and specific impacts?
- Detailed mapping table - Does the response have a detailed table mapping these causes and effects?
- Nuanced discussion - Does the response have detailed exploration of the topic and explicit discussion?
</Insightfulness Rules>

- Each section should follow the Helpfulness Rules.

<Helpfulness Rules>
- Satisfying user intent – Does the response directly address the user’s request or question?
- Ease of understanding – Is the response fluent, coherent, and logically structured?
- Accuracy – Are the facts, reasoning, and explanations correct?
- Appropriate language – Is the tone suitable and professional, without unnecessary jargon or confusing phrasing?
</Helpfulness Rules>

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- Include the URL in ### Sources section only. Use the citation number in the other sections.
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""


report_generation_with_draft_insight_prompt = """Based on all the research conducted and draft report, create a comprehensive, well-structured answer to the overall research brief:
<Research Brief>
{research_brief}
</Research Brief>

CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.
This is critical. The user will only understand the answer if it is written in the same language as their input message.

Today's date is {date}.

Here is the draft report:
<Draft Report>
{draft_report}
</Draft Report>

Here are the findings from the research that you conducted:
<Findings>
{findings}
</Findings>

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Keep important details from the research findings
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""