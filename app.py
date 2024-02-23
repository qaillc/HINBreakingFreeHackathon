import streamlit as st
import requests
import json
import os
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
import re
from PIL import Image
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt

from textwrap import dedent
import google.generativeai as genai

api_key = os.environ["OPENAI_API_KEY"]

from openai import OpenAI
import numpy as np
# Assuming chromadb and TruLens are correctly installed and configured
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens_eval import Tru, Feedback, Select, TruCustomApp
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
tru = Tru()


# Tool import
from crewai.tools.gemini_tools import GeminiSearchTools
from crewai.tools.mixtral_tools import MixtralSearchTools
from crewai.tools.zephyr_tools import ZephyrSearchTools
from crewai.tools.phi2_tools import Phi2SearchTools


# Google Langchain
from langchain_google_genai import GoogleGenerativeAI

#Crew imports
from crewai import Agent, Task, Crew, Process

# Retrieve API Key from Environment Variable
GOOGLE_AI_STUDIO = os.environ.get('GOOGLE_API_KEY')

# Ensure the API key is available
if not GOOGLE_AI_STUDIO:
    raise ValueError("API key not found. Please set the GOOGLE_AI_STUDIO2 environment variable.")

# Set gemini_llm
gemini_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO)


# Questions ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Scene 1: Morning Routine
question1 = """How does Alex use technology in his morning routine?"""

#Scene 2: Commute to Work
question2 = """What does Alex think about on his way to work?"""

#Scene 3: At Work
question3 = """How does working with his team affect Alex's work?"""

# Scene 4: Evening Relaxation
question4 = """What does Alex do to relax after work?"""

#Scene 5: Nighttime Wind-down
question5 = """How does Alex get ready for the next day before going to sleep?"""



# Bullets ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


text1_bullets ="""Alex, a software engineer in his 30s.

Scene 1: Morning Routine
Alex wakes up in his cluttered apartment, surrounded by tech gadgets and half-finished projects.
He brews his morning coffee with a smart coffee maker, the first gadget he ever programmed.
Alex checks his emails and calendar on a custom-built PC, planning his day ahead.
He does a quick 20-minute workout following a VR fitness program.
Before leaving, Alex feeds his cat, Pixel, and tells his smart home system to switch to energy-saving mode.
Scene 2: Commute to Work
Alex rides his electric bike through the bustling city streets, admiring the mix of old and new architecture.
He stops at his favorite local café, where the barista knows his order by heart: a double espresso to go.
Alex navigates through the morning rush, observing people and imagining what software could improve their lives.
He arrives at the tech park, parking his bike at the charging station.
Walking into his office building, Alex greets the security guards and staff, appreciating the community feeling at work.
Scene 3: At Work
Alex joins the morning stand-up meeting with his team, discussing the day's objectives and challenges.
He dives into coding, losing track of time as he debugs and develops new features for their project.
Lunchtime is a social hour, where Alex and his colleagues share ideas and talk about the latest tech trends over food from the cafeteria.
In the afternoon, Alex has a brainstorming session with the design team to discuss user interface improvements.
The day ends with Alex feeling a sense of accomplishment, having solved a particularly tricky problem that had been bugging the team for weeks.
Scene 4: Evening Relaxation
After work, Alex visits a local park, taking a moment to disconnect and enjoy nature.
He meets a friend for a casual dinner at a small but popular restaurant they've been wanting to try.
They discuss life, technology, and future plans, with Alex sharing his dream of starting his own tech venture.
On his way home, Alex stops by a bookstore, picking up a novel and a magazine about robotics.
Back at home, Alex spends some time reading and sketching ideas in his notebook, feeling inspired by the day's conversations.
Scene 5: Nighttime Wind-down
Alex prepares for bed by setting his smart home to night mode, dimming the lights and playing ambient sounds.
He checks his projects one last time, making a to-do list for the next day.
Alex spends a few minutes meditating, reflecting on the day and calming his mind.
He jots down a few thoughts in his journal, a habit he's kept for years to track his personal and professional growth.
As he falls asleep, Alex feels grateful for the day's experiences and optimistic about the future.

"""

# 5 Scene Stories ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Custom GPT Story

# OpenAI

text1 = """Five Scenes
    
Scene 1: Morning Routine

Alex's morning began in the familiar chaos of his cluttered apartment, where every surface was a testament to his life as a software engineer. Gadgets, both completed and in various stages of assembly, littered the space. His first act of the day, after silencing the alarm on his custom-programmed smartwatch, was to head straight for the kitchen. There, his smart coffee maker, the very first gadget he had ever tinkered with, awaited his command. As it whirred to life, Alex couldn't help but feel a sense of pride. This was a daily reminder of where it all started.

While sipping on his freshly brewed coffee, Alex booted up his custom-built PC. The dual monitors flickered to life, displaying his meticulously organized calendar and an inbox that was surprisingly manageable. This quiet time, before the world demanded his attention, was when he felt most at ease, plotting out his day against the backdrop of early morning silence.

Despite the digital demands of his life, Alex made it a point to care for his physical well-being. A 20-minute high-intensity workout in virtual reality not only got his blood pumping but also allowed him a brief escape into fantastical landscapes far removed from the urban sprawl of his reality.

Before leaving, he attended to Pixel, his aptly named cat, who was as much a fixture in his life as his love for technology. After ensuring she was fed, Alex gave a command to his smart home system, switching it to energy-saving mode. It was his small nod to sustainability, a principle he tried to incorporate into his life despite the energy-hungry demands of his profession.

Transition to Scene 2

With his day off to a structured start, Alex stepped out into the world, ready to face whatever challenges and surprises lay ahead. His commute to work was not just a necessary routine but a bridge between his personal sanctuary and the collaborative world of tech that awaited him.

Scene 2: Commute to Work

The journey to work was an electric glide through the city on his bike, a piece of tech that represented the perfect blend of efficiency and environmental consciousness. The city, with its contrasting architecture, always gave Alex food for thought. He saw it as a living, breathing entity, much like the code he worked with—constantly evolving, sometimes unpredictable, but always fascinating.

His stop at the local café was a ritual as much as a necessity. The familiarity of the place, the warmth of the greeting from the barista, and the perfection of the double espresso he ordered—it all contributed to the sense of community Alex cherished. In a world where digital connections often overshadowed the physical, these moments were precious.

Navigating through the morning rush, Alex observed the people around him. Each person was a story, a potential user of the future technologies he dreamed of creating. These observations often sparked ideas, some of which would find their way into his projects.

The tech park, with its sleek buildings and the promise of innovation, was the culmination of his morning journey. Parking his bike and walking into the building, Alex felt a surge of energy. The friendly faces of the security staff and office workers were reminders of the human element in technology, a factor that was too important to overlook.

Transition to Scene 3

As he settled into his workspace, surrounded by the buzz of activity and the familiar glow of computer screens, Alex knew he was exactly where he belonged. The transition from the solitary contemplation of his morning routine to the collaborative dynamism of his work environment was seamless, each phase of his day a vital component of his life as a software engineer.

Scene 3: At Work

The morning stand-up meeting was a whirlwind of updates and ideas. Alex and his team, a tight-knit group of talented individuals, shared their progress and challenges with a camaraderie that made even the toughest projects seem manageable. For Alex, these meetings were a reminder of the collective brainpower that drove their success.

Immersing himself in code, Alex entered a state of flow where hours passed like minutes. The act of debugging, the thrill of creating something new, the frustration of hitting a roadblock, and the euphoria of finally breaking through—it was a rollercoaster of emotions that he wouldn't trade for anything.

Lunchtime was an opportunity to step away from the screens and connect with his colleagues on a more personal level. Their conversations, ranging from the latest tech gadgets to speculative discussions about the future of software, were as nourishing as the food they shared.

The afternoon brainstorming session with the design team was a highlight. Alex's ability to translate complex technical concepts into user-friendly interfaces was one of his strengths. Collaborating with creatives who had a different perspective on technology was both challenging and rewarding.

Ending the day on a high note, having solved a problem that had been a thorn in the team's side, Alex felt a deep sense of satisfaction. It was moments like these that reaffirmed his love for his profession.

Transition to Scene 4

Leaving the office behind, Alex stepped out into the evening, his mind still buzzing with ideas but also ready to embrace the change of pace that the rest of his day promised. The transition from the structured environment of his workplace to the spontaneous possibilities of the evening was something he looked forward to.

Scene 4: Evening Relaxation

The park was an oasis of tranquility in the midst of the city's hustle. Here, Alex found a moment of peace, a brief period to just be and to recharge. The contrast between the natural surroundings and the digital world he inhabited was stark, yet Alex found beauty and value in both.

Dinner with a friend was a cherished ritual. Their conversation, free-flowing and wide-ranging, touched on topics that mattered to both of them. Alex's dreams of starting his own venture, the challenges and opportunities that lay ahead, were shared openly. It was a testament to the strength of their friendship that they could discuss their aspirations and fears with such honesty.

The bookstore visit was a spontaneous decision, but one that brought Alex a different kind of joy. Holding a book, flipping through its pages, felt almost revolutionary in a world dominated by screens. The novel and robotics magazine he chose were not just purchases; they were gateways to new worlds and ideas.

Returning home, Alex allowed himself the luxury of time—to read, to sketch, and to simply think. It was these moments of solitude and reflection that often sparked his most creative ideas.

Transition to Scene 5

As the day drew to a close, Alex prepared to transition from the world of wakefulness to the restorative embrace of sleep. The rituals that marked this final phase of his day were as important as those that began it, each step a preparation for the next day's adventures.

Scene 5: Nighttime Wind-down

Alex's home, now quiet and dimly lit, felt like a sanctuary as he activated the night mode on his smart home system. The soft ambient sounds that filled the space were designed to soothe and relax, a counterpoint to the day's stimulation.

Reviewing his projects and preparing a to-do list for the next day allowed Alex to organize his thoughts and priorities. It was a practice that brought both closure to the current day and a roadmap for the one to follow.

Meditation, a recent addition to his nightly routine, offered Alex a chance to clear his mind and center himself. It was a quiet rebellion against the constant noise of information and deadlines, a space he carved out for inner peace.

Journaling, a habit cultivated over years, was Alex's way of capturing the day's experiences, lessons learned, and moments of gratitude. It was a personal archive, a narrative of his journey through both his career and life.

As sleep claimed him, Alex felt a deep sense of gratitude for the day that had passed and a quiet anticipation for the one to come. His life, a blend of technology and humanity, challenges and achievements, was a journey he cherished deeply.


"""

text1_scene1 = """Scene 1: Morning Routine
Alex's morning began in the familiar chaos of his cluttered apartment, where every surface was a testament to his life as a software engineer. Gadgets, both completed and in various stages of assembly, littered the space. His first act of the day, after silencing the alarm on his custom-programmed smartwatch, was to head straight for the kitchen. There, his smart coffee maker, the very first gadget he had ever tinkered with, awaited his command. As it whirred to life, Alex couldn't help but feel a sense of pride. This was a daily reminder of where it all started.
While sipping on his freshly brewed coffee, Alex booted up his custom-built PC. The dual monitors flickered to life, displaying his meticulously organized calendar and an inbox that was surprisingly manageable. This quiet time, before the world demanded his attention, was when he felt most at ease, plotting out his day against the backdrop of early morning silence.
Despite the digital demands of his life, Alex made it a point to care for his physical well-being. A 20-minute high-intensity workout in virtual reality not only got his blood pumping but also allowed him a brief escape into fantastical landscapes far removed from the urban sprawl of his reality.
Before leaving, he attended to Pixel, his aptly named cat, who was as much a fixture in his life as his love for technology. After ensuring she was fed, Alex gave a command to his smart home system, switching it to energy-saving mode. It was his small nod to sustainability, a principle he tried to incorporate into his life despite the energy-hungry demands of his profession.
Transition to Scene 2
With his day off to a structured start, Alex stepped out into the world, ready to face whatever challenges and surprises lay ahead. His commute to work was not just a necessary routine but a bridge between his personal sanctuary and the collaborative world of tech that awaited him.
"""
text1_scene2 = """Scene 2: Commute to Work
The journey to work was an electric glide through the city on his bike, a piece of tech that represented the perfect blend of efficiency and environmental consciousness. The city, with its contrasting architecture, always gave Alex food for thought. He saw it as a living, breathing entity, much like the code he worked with—constantly evolving, sometimes unpredictable, but always fascinating.
His stop at the local café was a ritual as much as a necessity. The familiarity of the place, the warmth of the greeting from the barista, and the perfection of the double espresso he ordered—it all contributed to the sense of community Alex cherished. In a world where digital connections often overshadowed the physical, these moments were precious.
Navigating through the morning rush, Alex observed the people around him. Each person was a story, a potential user of the future technologies he dreamed of creating. These observations often sparked ideas, some of which would find their way into his projects.
The tech park, with its sleek buildings and the promise of innovation, was the culmination of his morning journey. Parking his bike and walking into the building, Alex felt a surge of energy. The friendly faces of the security staff and office workers were reminders of the human element in technology, a factor that was too important to overlook.
Transition to Scene 3
As he settled into his workspace, surrounded by the buzz of activity and the familiar glow of computer screens, Alex knew he was exactly where he belonged. The transition from the solitary contemplation of his morning routine to the collaborative dynamism of his work environment was seamless, each phase of his day a vital component of his life as a software engineer.
"""
text1_scene3 = """Scene 3: At Work
The morning stand-up meeting was a whirlwind of updates and ideas. Alex and his team, a tight-knit group of talented individuals, shared their progress and challenges with a camaraderie that made even the toughest projects seem manageable. For Alex, these meetings were a reminder of the collective brainpower that drove their success.
Immersing himself in code, Alex entered a state of flow where hours passed like minutes. The act of debugging, the thrill of creating something new, the frustration of hitting a roadblock, and the euphoria of finally breaking through—it was a rollercoaster of emotions that he wouldn't trade for anything.
Lunchtime was an opportunity to step away from the screens and connect with his colleagues on a more personal level. Their conversations, ranging from the latest tech gadgets to speculative discussions about the future of software, were as nourishing as the food they shared.
The afternoon brainstorming session with the design team was a highlight. Alex's ability to translate complex technical concepts into user-friendly interfaces was one of his strengths. Collaborating with creatives who had a different perspective on technology was both challenging and rewarding.
Ending the day on a high note, having solved a problem that had been a thorn in the team's side, Alex felt a deep sense of satisfaction. It was moments like these that reaffirmed his love for his profession.
Transition to Scene 4
Leaving the office behind, Alex stepped out into the evening, his mind still buzzing with ideas but also ready to embrace the change of pace that the rest of his day promised. The transition from the structured environment of his workplace to the spontaneous possibilities of the evening was something he looked forward to.
"""
text1_scene4 = """Scene 4: Evening Relaxation
The park was an oasis of tranquility in the midst of the city's hustle. Here, Alex found a moment of peace, a brief period to just be and to recharge. The contrast between the natural surroundings and the digital world he inhabited was stark, yet Alex found beauty and value in both.
Dinner with a friend was a cherished ritual. Their conversation, free-flowing and wide-ranging, touched on topics that mattered to both of them. Alex's dreams of starting his own venture, the challenges and opportunities that lay ahead, were shared openly. It was a testament to the strength of their friendship that they could discuss their aspirations and fears with such honesty.
The bookstore visit was a spontaneous decision, but one that brought Alex a different kind of joy. Holding a book, flipping through its pages, felt almost revolutionary in a world dominated by screens. The novel and robotics magazine he chose were not just purchases; they were gateways to new worlds and ideas.
Returning home, Alex allowed himself the luxury of time—to read, to sketch, and to simply think. It was these moments of solitude and reflection that often sparked his most creative ideas.
Transition to Scene 5
As the day drew to a close, Alex prepared to transition from the world of wakefulness to the restorative embrace of sleep. The rituals that marked this final phase of his day were as important as those that began it, each step a preparation for the next day's adventures.
"""
text1_scene5 = """Scene 5: Nighttime Wind-down
Alex's home, now quiet and dimly lit, felt like a sanctuary as he activated the night mode on his smart home system. The soft ambient sounds that filled the space were designed to soothe and relax, a counterpoint to the day's stimulation.
Reviewing his projects and preparing a to-do list for the next day allowed Alex to organize his thoughts and priorities. It was a practice that brought both closure to the current day and a roadmap for the one to follow.
Meditation, a recent addition to his nightly routine, offered Alex a chance to clear his mind and center himself. It was a quiet rebellion against the constant noise of information and deadlines, a space he carved out for inner peace.
Journaling, a habit cultivated over years, was Alex's way of capturing the day's experiences, lessons learned, and moments of gratitude. It was a personal archive, a narrative of his journey through both his career and life.
As sleep claimed him, Alex felt a deep sense of gratitude for the day that had passed and a quiet anticipation for the one to come. His life, a blend of technology and humanity, challenges and achievements, was a journey he cherished deeply.

"""

# Gemini Story
    
text2 = """In the heart of the city where technology and innovation thrived, Alex, a software engineer in his 30s, began his day with enthusiasm and a passion for creating extraordinary ideas.

Scene 1: Morning Routine

Alex woke up in his modern apartment, illuminated by the glow of digital displays. Gadgets of all kinds surrounded him, from the smart coffee maker brewing his morning brew to the VR fitness program guiding his workout. Amidst the organized chaos, Pixel, his feline companion, stretched and purred, seemingly energized by the day ahead. As Alex departed, his smart home system switched to energy-saving mode, leaving Pixel in a comfortable environment.

Scene 2: Commute to Work

Pedaling through the city streets on his electric bike, Alex relished the vibrant collision of old and new architecture. The city buzzed with life, and Alex couldn't help but imagine how technology could further enhance the urban experience. At his favorite café, the aroma of freshly roasted coffee filled the air as he enjoyed his regular double espresso. The barista, a familiar face, greeted him with a warm smile, knowing exactly how he liked his coffee.

Scene 3: At Work

Entering the tech park, Alex parked his bike at the charging station, feeling a sense of camaraderie as he greeted the security guards and staff. Inside his office, the morning stand-up meeting kicked off the day's activities. Surrounded by brilliant minds, Alex engaged in discussions about project objectives and challenges. Time seemed to dissolve as he immersed himself in coding, debugging, and developing new features. Lunchtime brought a reprieve, a chance to socialize and share ideas with colleagues over food from the bustling cafeteria. The afternoon brought a brainstorming session with the design team, where creativity flowed and ideas took shape.

Scene 4: Evening Relaxation

As the day drew to a close, Alex sought respite in the tranquility of the local park. The lush greenery and the gentle breeze soothed his mind, offering a much-needed escape from the digital realm. Meeting a friend for dinner at a cozy restaurant, Alex engaged in lively conversations about life, technology, and future aspirations. The shared laughter and heartfelt talks filled his heart with warmth. On his way home, he stopped by a bookstore, his footsteps echoing through the aisles as he browsed through novels and magazines.

Scene 5: Nighttime Wind-down

Returning home, Alex transitioned into relaxation mode. With a few taps on his smartphone, his smart home system dimmed the lights and filled the air with ambient sounds. A sense of accomplishment washed over him as he reviewed his work, making a to-do list for the following day. The day's experiences swirled in his mind as he jotted down thoughts in his journal, capturing his personal and professional growth. As he drifted into sleep, Alex felt a profound sense of gratitude for the day's events and an unwavering optimism for the future.

    
"""

# Gemini Scenes
text2_scene1 = """In the heart of the city where technology and innovation thrived, Alex, a software engineer in his 30s, began his day with enthusiasm and a passion for creating extraordinary ideas.
Scene 1: Morning Routine
Alex woke up in his modern apartment, illuminated by the glow of digital displays. Gadgets of all kinds surrounded him, from the smart coffee maker brewing his morning brew to the VR fitness program guiding his workout. Amidst the organized chaos, Pixel, his feline companion, stretched and purred, seemingly energized by the day ahead. As Alex departed, his smart home system switched to energy-saving mode, leaving Pixel in a comfortable environment.
"""
text2_scene2 = """Scene 2: Commute to Work
Pedaling through the city streets on his electric bike, Alex relished the vibrant collision of old and new architecture. The city buzzed with life, and Alex couldn't help but imagine how technology could further enhance the urban experience. At his favorite café, the aroma of freshly roasted coffee filled the air as he enjoyed his regular double espresso. The barista, a familiar face, greeted him with a warm smile, knowing exactly how he liked his coffee.
"""
text2_scene3 = """Scene 3: At Work
Entering the tech park, Alex parked his bike at the charging station, feeling a sense of camaraderie as he greeted the security guards and staff. Inside his office, the morning stand-up meeting kicked off the day's activities. Surrounded by brilliant minds, Alex engaged in discussions about project objectives and challenges. Time seemed to dissolve as he immersed himself in coding, debugging, and developing new features. Lunchtime brought a reprieve, a chance to socialize and share ideas with colleagues over food from the bustling cafeteria. The afternoon brought a brainstorming session with the design team, where creativity flowed and ideas took shape.
"""
text2_scene4 = """Scene 4: Evening Relaxation
As the day drew to a close, Alex sought respite in the tranquility of the local park. The lush greenery and the gentle breeze soothed his mind, offering a much-needed escape from the digital realm. Meeting a friend for dinner at a cozy restaurant, Alex engaged in lively conversations about life, technology, and future aspirations. The shared laughter and heartfelt talks filled his heart with warmth. On his way home, he stopped by a bookstore, his footsteps echoing through the aisles as he browsed through novels and magazines.
"""
text2_scene5 = """Scene 5: Nighttime Wind-down
Returning home, Alex transitioned into relaxation mode. With a few taps on his smartphone, his smart home system dimmed the lights and filled the air with ambient sounds. A sense of accomplishment washed over him as he reviewed his work, making a to-do list for the following day. The day's experiences swirled in his mind as he jotted down thoughts in his journal, capturing his personal and professional growth. As he drifted into sleep, Alex felt a profound sense of gratitude for the day's events and an unwavering optimism for the future.  
"""

# Mixtral Crazy Story
text3 = """ In the heart of a vast and vibrant metropolis, dwelled a man named Alex, a gifted software engineer navigating the intricate dance of life, technology, and human connection. Alex's days were a tapestry woven with the threads of innovation, camaraderie, and a relentless pursuit of progress.

Scene 1: Morning Symphony

In the hushed hours before dawn, as the city slumbered, Alex stirred from his slumber within his cluttered apartment, a sanctuary of technological marvels. Gadgets of all shapes and sizes adorned his living space, half-finished projects hinting at his boundless creativity. A soft glow emanated from his custom-built PC, the first gadget he had ever programmed, displaying his meticulously organized emails and calendar entries. The aroma of freshly brewed coffee filled the air, a symphony of flavors conjured by his smart coffee maker. With the grace of a seasoned conductor, Alex prepared his morning elixir, each step a testament to his mastery over technology.

After nourishing his body with caffeine, Alex embarked on a brisk 20-minute workout, guided by a VR fitness program. His movements flowed in harmony with the virtual instructor, every squat, lunge, and plank propelling him towards a state of physical and mental vitality. Before venturing out into the bustling city, Alex remembered his feline companion, Pixel, showering the cat with affection and ensuring his smart home system transitioned into energy-saving mode.

Scene 2: Urban Rhapsody

The city streets hummed with life as Alex navigated the urban labyrinth on his electric bike. Buildings of various architectural epochs lined the streets, a testament to the city's rich history and ever-evolving skyline. The morning sun cast long shadows, creating a tapestry of light and shadow that danced across his path. As he pedaled through the throngs of commuters, Alex couldn't help but marvel at the sheer diversity of humanity. Each individual represented a unique story, a microcosm of experiences and aspirations. His mind raced, conjuring possibilities for how technology could enhance their lives, bridging the gap between the physical and the digital.

In the midst of the urban symphony, Alex made a pit stop at his favorite local café, where the barista greeted him with a warm smile and a perfectly crafted double espresso to go. The aroma of freshly baked pastries mingled with the buzz of conversation, creating an ambiance that was both stimulating and comforting. As he sipped his coffee, Alex observed the people around him, their faces illuminated by the screens of their smartphones. It was a poignant reminder of the pervasive influence of technology in modern society, a force that had the potential to both connect and divide.

Scene 3: The Crucible of Innovation

The tech park was a hub of creativity and collaboration, a place where dreams were forged into reality. Alex arrived at his office, parking his bike at the charging station, feeling a surge of anticipation as he walked through the entrance. The security guards and staff greeted him with familiar smiles, fostering a sense of community and belonging. Inside the office, the air crackled with energy as teams gathered for their morning stand-up meetings, discussing the day's objectives and challenges with a shared sense of purpose.

Alex immersed himself in the depths of coding, his fingers dancing across the keyboard in a rhythmic symphony. Time seemed to slip away as he debugged and developed new features for his project, driven by an insatiable passion for innovation. Lunchtime brought a respite from the intensity of his work, a chance to socialize with colleagues and exchange ideas over food from the cafeteria. Laughter and camaraderie filled the air, a testament to the strong bonds that had been forged within the team.

In the afternoon, Alex joined forces with the design team for a brainstorming session, exploring user interface improvements that would enhance the user experience. Ideas flowed freely, bouncing off one another like sparks in a storm. By day's end, Alex felt a profound sense of accomplishment, having solved a particularly vexing problem that had plagued the team for weeks. The satisfaction of overcoming a technical hurdle was a powerful reminder of his skills and the impact he could make.

Scene 4: Eventide Reverie

As dusk descended, Alex sought solace in the tranquility of a nearby park, a green oasis amid the urban sprawl. He strolled along the winding paths, surrounded by the gentle rustling of leaves and the distant calls of birds. The park was a sanctuary, a place to disconnect from the demands of work and reconnect with the natural world. In this serene setting, Alex allowed his thoughts to wander, reflecting on the conversations and experiences of the day. The seeds of inspiration had been sown, and he felt a renewed sense of purpose.

Later, Alex met a friend for dinner at a small but popular restaurant that had been on their list to try. The atmosphere was warm and inviting, the aroma of culinary delights permeating the air. Over a shared meal, they delved into discussions about life, technology, and future plans. Alex shared his dream of starting his own tech venture, his eyes sparkling with passion and ambition. His friend listened intently, offering encouragement and support.

On his way home, Alex stopped by a bookstore, drawn by the allure of the written word. He carefully selected a novel and a magazine about robotics, eager to expand his knowledge and explore new perspectives. Back at his apartment, he settled into his favorite chair, the soft glow of a reading lamp illuminating the pages of the book. The world outside faded away as he immersed himself in the narratives, his imagination ignited by the power of storytelling.

Scene 5: Nocturnal Reflections

As the hour grew late, Alex prepared for bed, setting his smart home to night mode. The lights dimmed, and the gentle sounds of ambient music filled the room, creating a soothing atmosphere. He checked his projects one last time, making a meticulous to-do list for the next day. With a sense of order and purpose, he closed his laptop, ready to embark on the journey of sleep.

Before succumbing to slumber, Alex took a few moments to meditate, calming his mind and reflecting on the day's events. He jotted down his thoughts in his journal, a repository of his personal and professional growth. The flickering candlelight cast shadows on the pages, illuminating the words that traced his life's trajectory. As sleep enveloped him, Alex felt a profound sense of gratitude for the experiences of the day and an unwavering optimism for the future.
"""

# Mixtral Crazy Scenes
text3_scene1 = """ In the heart of a vast and vibrant metropolis, dwelled a man named Alex, a gifted software engineer navigating the intricate dance of life, technology, and human connection. Alex's days were a tapestry woven with the threads of innovation, camaraderie, and a relentless pursuit of progress.
Scene 1: Morning Symphony
In the hushed hours before dawn, as the city slumbered, Alex stirred from his slumber within his cluttered apartment, a sanctuary of technological marvels. Gadgets of all shapes and sizes adorned his living space, half-finished projects hinting at his boundless creativity. A soft glow emanated from his custom-built PC, the first gadget he had ever programmed, displaying his meticulously organized emails and calendar entries. The aroma of freshly brewed coffee filled the air, a symphony of flavors conjured by his smart coffee maker. With the grace of a seasoned conductor, Alex prepared his morning elixir, each step a testament to his mastery over technology.
After nourishing his body with caffeine, Alex embarked on a brisk 20-minute workout, guided by a VR fitness program. His movements flowed in harmony with the virtual instructor, every squat, lunge, and plank propelling him towards a state of physical and mental vitality. Before venturing out into the bustling city, Alex remembered his feline companion, Pixel, showering the cat with affection and ensuring his smart home system transitioned into energy-saving mode.
"""
text3_scene2 = """Scene 2: Urban Rhapsody
The city streets hummed with life as Alex navigated the urban labyrinth on his electric bike. Buildings of various architectural epochs lined the streets, a testament to the city's rich history and ever-evolving skyline. The morning sun cast long shadows, creating a tapestry of light and shadow that danced across his path. As he pedaled through the throngs of commuters, Alex couldn't help but marvel at the sheer diversity of humanity. Each individual represented a unique story, a microcosm of experiences and aspirations. His mind raced, conjuring possibilities for how technology could enhance their lives, bridging the gap between the physical and the digital.
In the midst of the urban symphony, Alex made a pit stop at his favorite local café, where the barista greeted him with a warm smile and a perfectly crafted double espresso to go. The aroma of freshly baked pastries mingled with the buzz of conversation, creating an ambiance that was both stimulating and comforting. As he sipped his coffee, Alex observed the people around him, their faces illuminated by the screens of their smartphones. It was a poignant reminder of the pervasive influence of technology in modern society, a force that had the potential to both connect and divide.
"""
text3_scene3 = """Scene 3: The Crucible of Innovation
The tech park was a hub of creativity and collaboration, a place where dreams were forged into reality. Alex arrived at his office, parking his bike at the charging station, feeling a surge of anticipation as he walked through the entrance. The security guards and staff greeted him with familiar smiles, fostering a sense of community and belonging. Inside the office, the air crackled with energy as teams gathered for their morning stand-up meetings, discussing the day's objectives and challenges with a shared sense of purpose.
Alex immersed himself in the depths of coding, his fingers dancing across the keyboard in a rhythmic symphony. Time seemed to slip away as he debugged and developed new features for his project, driven by an insatiable passion for innovation. Lunchtime brought a respite from the intensity of his work, a chance to socialize with colleagues and exchange ideas over food from the cafeteria. Laughter and camaraderie filled the air, a testament to the strong bonds that had been forged within the team.
In the afternoon, Alex joined forces with the design team for a brainstorming session, exploring user interface improvements that would enhance the user experience. Ideas flowed freely, bouncing off one another like sparks in a storm. By day's end, Alex felt a profound sense of accomplishment, having solved a particularly vexing problem that had plagued the team for weeks. The satisfaction of overcoming a technical hurdle was a powerful reminder of his skills and the impact he could make.
"""
text3_scene4 = """Scene 4: Eventide Reverie
As dusk descended, Alex sought solace in the tranquility of a nearby park, a green oasis amid the urban sprawl. He strolled along the winding paths, surrounded by the gentle rustling of leaves and the distant calls of birds. The park was a sanctuary, a place to disconnect from the demands of work and reconnect with the natural world. In this serene setting, Alex allowed his thoughts to wander, reflecting on the conversations and experiences of the day. The seeds of inspiration had been sown, and he felt a renewed sense of purpose.
Later, Alex met a friend for dinner at a small but popular restaurant that had been on their list to try. The atmosphere was warm and inviting, the aroma of culinary delights permeating the air. Over a shared meal, they delved into discussions about life, technology, and future plans. Alex shared his dream of starting his own tech venture, his eyes sparkling with passion and ambition. His friend listened intently, offering encouragement and support.
On his way home, Alex stopped by a bookstore, drawn by the allure of the written word. He carefully selected a novel and a magazine about robotics, eager to expand his knowledge and explore new perspectives. Back at his apartment, he settled into his favorite chair, the soft glow of a reading lamp illuminating the pages of the book. The world outside faded away as he immersed himself in the narratives, his imagination ignited by the power of storytelling.
"""
text3_scene5 = """Scene 5: Nocturnal Reflections
As the hour grew late, Alex prepared for bed, setting his smart home to night mode. The lights dimmed, and the gentle sounds of ambient music filled the room, creating a soothing atmosphere. He checked his projects one last time, making a meticulous to-do list for the next day. With a sense of order and purpose, he closed his laptop, ready to embark on the journey of sleep.
Before succumbing to slumber, Alex took a few moments to meditate, calming his mind and reflecting on the day's events. He jotted down his thoughts in his journal, a repository of his personal and professional growth. The flickering candlelight cast shadows on the pages, illuminating the words that traced his life's trajectory. As sleep enveloped him, Alex felt a profound sense of gratitude for the experiences of the day and an unwavering optimism for the future.
"""

# Mixtral Normal Story
text4 = """ In the bustling city of Technopolis, Alex, a software engineer with an inventive spirit, embarks on his daily routine, navigating a world shaped by technology and human connection.

Scene 1: Morning Routine

Dragging himself out of bed in his tech-filled apartment, Alex brewed his morning coffee using the smart coffee maker he once programmed. With the aroma of freshly brewed coffee filling the air, he settled down at his custom-built PC, checking emails and planning his day with the assistance of his digital calendar. As the sun peeked through the window, he immersed himself in a quick 20-minute VR fitness workout, getting his body and mind ready for the day ahead. Before leaving for work, Alex didn't forget his beloved cat, Pixel. He filled its bowl with food and instructed his smart home system to switch to energy-saving mode, ensuring a comfortable and efficient environment for his furry companion.

Scene 2: Commute to Work

The morning air was crisp as Alex pedaled his electric bike through the city streets, marveling at the blend of towering skyscrapers and charming old buildings. Making a brief stop at his favorite café, he was greeted by the friendly barista who knew his order by heart—a double espresso to kick-start his day. As he cycled through the bustling traffic, Alex observed the people around him, his mind buzzing with ideas for innovative software solutions that could improve their lives. Reaching the tech park, he parked his bike at the charging station, ready to delve into the day's challenges. Entering his office building, Alex exchanged warm greetings with the security guards and staff, appreciating the sense of community he felt among his colleagues.

Scene 3: At Work

The morning stand-up meeting brought Alex and his team together, where they discussed the day's goals and addressed any obstacles they might encounter. Immersing himself in the world of coding, time seemed to slip away as Alex debugged and developed new features for their project. Lunchtime was a delightful social hour, filled with laughter and lively conversations about tech trends and personal anecdotes. In the afternoon, Alex collaborated with the design team in a brainstorming session, exploring creative user interface improvements that would enhance the user experience. As the day drew to a close, Alex felt a profound sense of satisfaction, having solved a particularly vexing problem that had puzzled the team for weeks.

Scene 4: Evening Relaxation

Stepping out of the office, Alex headed to a nearby park, seeking solace and inspiration in nature's embrace. The lush greenery and tranquil atmosphere helped him disconnect from the digital world and reconnect with his inner self. Later, he met a friend for a relaxed dinner at a charming restaurant they had been eager to try. Over plates of delicious food, they engaged in thought-provoking discussions about life, technology, and their dreams for the future. Alex shared his aspiration of launching his own tech venture, his eyes gleaming with passion and determination. On his way home, he made a detour to a bookstore, drawn by the allure of a captivating novel and a magazine dedicated to the fascinating world of robotics. Back in the comfort of his apartment, Alex delved into the pages of the novel and sketched ideas in his notebook, his mind abuzz with creativity sparked by the day's experiences.

Scene 5: Nighttime Wind-down

As the day drew to a close, Alex prepared for a restful night's sleep. He activated his smart home's night mode, dimming the lights and filling the room with soothing ambient sounds. With meticulous care, he reviewed his projects, making a to-do list for the next day to ensure he stayed organized and productive. To quiet his mind and reflect on the day's events, Alex spent a few moments in meditation, finding solace in the stillness. Before slipping into slumber, he jotted down his thoughts and reflections in his journal, a practice he had maintained for years to capture his personal and professional growth. As he drifted off to sleep, Alex felt a deep sense of gratitude for the day's experiences and an unwavering optimism for the future, knowing that each new day held endless possibilities for innovation and human connection.

"""

# Mixtral Normal Scenes
text4_scene1 = """ In the bustling city of Technopolis, Alex, a software engineer with an inventive spirit, embarks on his daily routine, navigating a world shaped by technology and human connection.
Scene 1: Morning Routine
Dragging himself out of bed in his tech-filled apartment, Alex brewed his morning coffee using the smart coffee maker he once programmed. With the aroma of freshly brewed coffee filling the air, he settled down at his custom-built PC, checking emails and planning his day with the assistance of his digital calendar. As the sun peeked through the window, he immersed himself in a quick 20-minute VR fitness workout, getting his body and mind ready for the day ahead. Before leaving for work, Alex didn't forget his beloved cat, Pixel. He filled its bowl with food and instructed his smart home system to switch to energy-saving mode, ensuring a comfortable and efficient environment for his furry companion.
"""
text4_scene2 = """Scene 2: Commute to Work
The morning air was crisp as Alex pedaled his electric bike through the city streets, marveling at the blend of towering skyscrapers and charming old buildings. Making a brief stop at his favorite café, he was greeted by the friendly barista who knew his order by heart—a double espresso to kick-start his day. As he cycled through the bustling traffic, Alex observed the people around him, his mind buzzing with ideas for innovative software solutions that could improve their lives. Reaching the tech park, he parked his bike at the charging station, ready to delve into the day's challenges. Entering his office building, Alex exchanged warm greetings with the security guards and staff, appreciating the sense of community he felt among his colleagues.
"""
text4_scene3 = """Scene 3: At Work
The morning stand-up meeting brought Alex and his team together, where they discussed the day's goals and addressed any obstacles they might encounter. Immersing himself in the world of coding, time seemed to slip away as Alex debugged and developed new features for their project. Lunchtime was a delightful social hour, filled with laughter and lively conversations about tech trends and personal anecdotes. In the afternoon, Alex collaborated with the design team in a brainstorming session, exploring creative user interface improvements that would enhance the user experience. As the day drew to a close, Alex felt a profound sense of satisfaction, having solved a particularly vexing problem that had puzzled the team for weeks.
"""
text4_scene4 = """Scene 4: Evening Relaxation
Stepping out of the office, Alex headed to a nearby park, seeking solace and inspiration in nature's embrace. The lush greenery and tranquil atmosphere helped him disconnect from the digital world and reconnect with his inner self. Later, he met a friend for a relaxed dinner at a charming restaurant they had been eager to try. Over plates of delicious food, they engaged in thought-provoking discussions about life, technology, and their dreams for the future. Alex shared his aspiration of launching his own tech venture, his eyes gleaming with passion and determination. On his way home, he made a detour to a bookstore, drawn by the allure of a captivating novel and a magazine dedicated to the fascinating world of robotics. Back in the comfort of his apartment, Alex delved into the pages of the novel and sketched ideas in his notebook, his mind abuzz with creativity sparked by the day's experiences.
"""
text4_scene5 = """Scene 5: Nighttime Wind-down
As the day drew to a close, Alex prepared for a restful night's sleep. He activated his smart home's night mode, dimming the lights and filling the room with soothing ambient sounds. With meticulous care, he reviewed his projects, making a to-do list for the next day to ensure he stayed organized and productive. To quiet his mind and reflect on the day's events, Alex spent a few moments in meditation, finding solace in the stillness. Before slipping into slumber, he jotted down his thoughts and reflections in his journal, a practice he had maintained for years to capture his personal and professional growth. As he drifted off to sleep, Alex felt a deep sense of gratitude for the day's experiences and an unwavering optimism for the future, knowing that each new day held endless possibilities for innovation and human connection.
"""

# Zephra Story
text5 = """In the heart of a bustling metropolis, amidst towering skyscrapers and vibrant streets, resided Alex, a software engineer in his thirties, whose life was deeply entwined with technology.

Scene 1: Morning Routine

Alex awoke to the gentle hum of his smart alarm clock, a symphony of beeps and boops that gently coaxed him from the realm of dreams. As he stirred from slumber, his eyes beheld a symphony of tech gadgets and unfinished projects strewn across his cluttered apartment. Amidst this technological tapestry, Alex performed his morning ablutions, his movements guided by the rhythmic beeping of his toothbrush and the soothing whir of his electric shaver.

Transition: As the sun cast its golden rays through the cityscape, Alex embarked on his daily commute, a journey that wove its way through the bustling streets, a tapestry of human activity and architectural wonders.

Scene 2: Commute to Work

Alex straddled his electric bike, the motor purring beneath him as he navigated the labyrinthine streets. The city hummed with life, a symphony of sounds that echoed off the towering skyscrapers and reverberated through the narrow alleyways. Alex paused at his favorite local café, where the aroma of freshly brewed coffee filled the air, a welcome respite from the cacophony of the city. With a nod and a smile, the barista handed Alex his double espresso, a perfect blend of caffeine and warmth that energized his spirit.

Transition: The day unfolded before Alex like a tapestry of challenges and opportunities. He arrived at the tech park, a hub of innovation and creativity, and parked his bike at the charging station, a symbol of the seamless integration of technology into his life.

Scene 3: At Work

Alex entered his office building, greeted by the familiar faces of security guards and staff, a community brought together by their shared passion for technology. The morning stand-up meeting buzzed with excitement as team members discussed the day's objectives and shared ideas. Alex immersed himself in the creative flow of coding, the rhythmic tapping of his fingers on the keyboard a testament to his dedication. Lunchtime became a social hour, a time to connect with colleagues, exchange insights, and explore the latest tech trends. In the afternoon, Alex collaborated with the design team, brainstorming user interface improvements, his mind ablaze with innovative possibilities.

Transition: As the day drew to a close, Alex felt a sense of accomplishment, a satisfaction born of solving a particularly tricky problem that had plagued the team for weeks. With a weary yet fulfilled smile, he bid farewell to his colleagues and stepped out into the evening air.

Scene 4: Evening Relaxation

Alex sought solace in nature, seeking a respite from the constant buzz of technology. He wandered through the local park, the gentle rustle of leaves underfoot and the sweet scent of blooming flowers soothing his senses. Later, he met a friend for dinner at a cozy restaurant, where they engaged in lively conversation, exploring the intersection of life, technology, and future aspirations. On his way home, Alex stopped by a bookstore, drawn to the shelves filled with novels and magazines that promised to expand his knowledge and ignite his imagination.

Transition: As night fell, Alex returned to his apartment, a sanctuary of technology and personal growth. He prepared for bed, dimming the lights and setting his smart home to night mode, creating an atmosphere conducive to relaxation and reflection.

Scene 5: Nighttime Wind-down

With a sense of tranquility, Alex reviewed his projects one last time, making a meticulous to-do list for the next day. He spent a few minutes in meditation, allowing his thoughts to drift away, finding inner peace amidst the digital symphony that surrounded him. Before retiring to bed, Alex jotted down his thoughts in his journal, a chronicle of his personal and professional journey. As he drifted off to sleep, Alex felt a profound sense of gratitude for the day's experiences, knowing that tomorrow would bring new challenges, new opportunities, and new possibilities.
    
"""

#Zephra Scenes
text5_scene1 = """ In the heart of a bustling metropolis, amidst towering skyscrapers and vibrant streets, resided Alex, a software engineer in his thirties, whose life was deeply entwined with technology.
Scene 1: Morning Routine
Alex awoke to the gentle hum of his smart alarm clock, a symphony of beeps and boops that gently coaxed him from the realm of dreams. As he stirred from slumber, his eyes beheld a symphony of tech gadgets and unfinished projects strewn across his cluttered apartment. Amidst this technological tapestry, Alex performed his morning ablutions, his movements guided by the rhythmic beeping of his toothbrush and the soothing whir of his electric shaver.
Transition: As the sun cast its golden rays through the cityscape, Alex embarked on his daily commute, a journey that wove its way through the bustling streets, a tapestry of human activity and architectural wonders.
"""
text5_scene2 = """ Scene 2: Commute to Work
Alex straddled his electric bike, the motor purring beneath him as he navigated the labyrinthine streets. The city hummed with life, a symphony of sounds that echoed off the towering skyscrapers and reverberated through the narrow alleyways. Alex paused at his favorite local café, where the aroma of freshly brewed coffee filled the air, a welcome respite from the cacophony of the city. With a nod and a smile, the barista handed Alex his double espresso, a perfect blend of caffeine and warmth that energized his spirit.
Transition: The day unfolded before Alex like a tapestry of challenges and opportunities. He arrived at the tech park, a hub of innovation and creativity, and parked his bike at the charging station, a symbol of the seamless integration of technology into his life.
"""
text5_scene3 = """ Scene 3: At Work
Alex entered his office building, greeted by the familiar faces of security guards and staff, a community brought together by their shared passion for technology. The morning stand-up meeting buzzed with excitement as team members discussed the day's objectives and shared ideas. Alex immersed himself in the creative flow of coding, the rhythmic tapping of his fingers on the keyboard a testament to his dedication. Lunchtime became a social hour, a time to connect with colleagues, exchange insights, and explore the latest tech trends. In the afternoon, Alex collaborated with the design team, brainstorming user interface improvements, his mind ablaze with innovative possibilities.
Transition: As the day drew to a close, Alex felt a sense of accomplishment, a satisfaction born of solving a particularly tricky problem that had plagued the team for weeks. With a weary yet fulfilled smile, he bid farewell to his colleagues and stepped out into the evening air.
"""
text5_scene4 = """ Scene 4: Evening Relaxation
Alex sought solace in nature, seeking a respite from the constant buzz of technology. He wandered through the local park, the gentle rustle of leaves underfoot and the sweet scent of blooming flowers soothing his senses. Later, he met a friend for dinner at a cozy restaurant, where they engaged in lively conversation, exploring the intersection of life, technology, and future aspirations. On his way home, Alex stopped by a bookstore, drawn to the shelves filled with novels and magazines that promised to expand his knowledge and ignite his imagination.
Transition: As night fell, Alex returned to his apartment, a sanctuary of technology and personal growth. He prepared for bed, dimming the lights and setting his smart home to night mode, creating an atmosphere conducive to relaxation and reflection.
"""
text5_scene5 = """ Scene 5: Nighttime Wind-down
With a sense of tranquility, Alex reviewed his projects one last time, making a meticulous to-do list for the next day. He spent a few minutes in meditation, allowing his thoughts to drift away, finding inner peace amidst the digital symphony that surrounded him. Before retiring to bed, Alex jotted down his thoughts in his journal, a chronicle of his personal and professional journey. As he drifted off to sleep, Alex felt a profound sense of gratitude for the day's experiences, knowing that tomorrow would bring new challenges, new opportunities, and new possibilities.
"""


# Phi-2 Story
text6 = """ In the heart of the bustling city, Alex, a software engineer with a passion for innovation, embarked on a typical day filled with technological marvels and human connections.

Scene 1: Morning Routine Awakening to the gentle hum of his smart alarm clock, Alex stirred from his slumber, his bleary eyes adjusting to the soft glow of the digital display. As he stretched his limbs, the aroma of freshly brewed coffee wafted from the adjacent kitchen, beckoning him to start his day.

Scene 2: Commute to Work With his trusty electric bike gliding silently through the city's vibrant tapestry of old and new architecture, Alex felt a sense of exhilaration as he navigated through the morning rush. Pausing at his favorite local café, he exchanged warm greetings with the barista, who seemed to know his order by heart – a double espresso to jolt him into gear for the day ahead.

Scene 3: At Work Stepping into the bustling office building, Alex was greeted by the friendly smiles of the security guards and staff, creating a welcoming atmosphere that made him feel like part of a close-knit community. The morning stand-up meeting with his team was a hive of activity, with lively discussions about the day's objectives and challenges, setting the tone for a productive and collaborative workday.

Scene 4: Evening Relaxation As the sun began its descent, Alex sought solace in the tranquility of a nearby park, immersing himself in the beauty of nature and disconnecting from the digital realm. Later, he met a friend for a casual dinner at a cozy restaurant, where they delved into conversations about life, technology, and future aspirations.

Scene 5: Nighttime Wind-down Returning to the solitude of his apartment, Alex indulged in a calming nighttime routine. He dimmed the lights, engaged ambient sounds to soothe his senses, and jotted down his thoughts in a journal, a practice that helped him reflect on the day's experiences and track his personal growth. As he lay down on his bed, a sense of contentment washed over him, knowing that he was part of something bigger than himself – a world where technology and human ingenuity intertwined to create a tapestry of innovation and progress.
     
"""
# Phi-2 Scenes
text6_scene1 = """ In the heart of the bustling city, Alex, a software engineer with a passion for innovation, embarked on a typical day filled with technological marvels and human connections.
Scene 1: Morning Routine Awakening to the gentle hum of his smart alarm clock, Alex stirred from his slumber, his bleary eyes adjusting to the soft glow of the digital display. As he stretched his limbs, the aroma of freshly brewed coffee wafted from the adjacent kitchen, beckoning him to start his day.
"""
text6_scene2 = """ Scene 2: Commute to Work With his trusty electric bike gliding silently through the city's vibrant tapestry of old and new architecture, Alex felt a sense of exhilaration as he navigated through the morning rush. Pausing at his favorite local café, he exchanged warm greetings with the barista, who seemed to know his order by heart – a double espresso to jolt him into gear for the day ahead.
"""
text6_scene3 = """ Scene 3: At Work Stepping into the bustling office building, Alex was greeted by the friendly smiles of the security guards and staff, creating a welcoming atmosphere that made him feel like part of a close-knit community. The morning stand-up meeting with his team was a hive of activity, with lively discussions about the day's objectives and challenges, setting the tone for a productive and collaborative workday.
"""
text6_scene4 = """ Scene 4: Evening Relaxation As the sun began its descent, Alex sought solace in the tranquility of a nearby park, immersing himself in the beauty of nature and disconnecting from the digital realm. Later, he met a friend for a casual dinner at a cozy restaurant, where they delved into conversations about life, technology, and future aspirations.
"""
text6_scene5 = """ Scene 5: Nighttime Wind-down Returning to the solitude of his apartment, Alex indulged in a calming nighttime routine. He dimmed the lights, engaged ambient sounds to soothe his senses, and jotted down his thoughts in a journal, a practice that helped him reflect on the day's experiences and track his personal growth. As he lay down on his bed, a sense of contentment washed over him, knowing that he was part of something bigger than himself – a world where technology and human ingenuity intertwined to create a tapestry of innovation and progress.
"""


# Custom GPT Image Scene Description 

text1_image = """Image Descriptions
    
Scene 1: Morning Routine

Alex wakes up in his cluttered apartment, surrounded by tech gadgets and half-finished projects. The room is filled with various screens, books, and a smart coffee maker on a kitchen counter. Alex, a man in his 30s with short hair, is stretching and yawning, ready to start his day. The early morning light filters through the window, casting a soft glow over the chaos of innovation and creativity that defines his living space.

Scene 2: Commute to Work

Alex rides his electric bike through the bustling city streets, admiring the mix of old and new architecture. The scene captures a dynamic urban environment with skyscrapers, historic buildings, and busy pedestrians. Alex, wearing a helmet and casual work attire, navigates his bike with ease, reflecting a connection between technology and traditional city life. The morning sun illuminates the city, highlighting the energy of the start of a new day.

Scene 3: At Work

Alex is immersed in his work, surrounded by multiple monitors displaying code and software development tools. The office environment is lively, with colleagues collaborating in the background. Alex, focused and engaged, types away on his keyboard, solving complex problems. The scene conveys a sense of dedication and passion for technology, with notes and diagrams scattered around his workspace, and a cup of coffee nearby to keep him energized.

Scene 4: Evening Relaxation

Alex enjoys a peaceful moment in a lush, green park, sitting on a bench with a book in hand. The park is filled with tall trees, colorful flowers, and a small pond reflecting the sky. People are walking dogs and jogging in the background, creating a serene and lively atmosphere. Alex, taking a break from technology, is relaxed and content, absorbed in his novel. The setting sun casts a warm glow over the scene, enhancing the tranquility and beauty of the moment.

Scene 5: Nighttime Wind-down

Alex prepares for bed in his smart home, which is set to night mode with dim lights and ambient sounds. The scene shows a modern bedroom with a comfortable bed, smart gadgets around, and a window showing the night sky. Alex, in comfortable nightwear, is jotting down notes in a journal, reflecting on the day. The room exudes a calm and cozy atmosphere, with a book and a meditation mat nearby, highlighting his routine of winding down and gathering thoughts before sleep.
    
"""

# Custom GPT Image Scene Description 

text1_image_scene1 = """Image Descriptions 
Scene 1: Morning Routine
Alex wakes up in his cluttered apartment, surrounded by tech gadgets and half-finished projects. The room is filled with various screens, books, and a smart coffee maker on a kitchen counter. Alex, a man in his 30s with short hair, is stretching and yawning, ready to start his day. The early morning light filters through the window, casting a soft glow over the chaos of innovation and creativity that defines his living space.
"""
text1_image_scene2 = """Scene 2: Commute to Work
Alex rides his electric bike through the bustling city streets, admiring the mix of old and new architecture. The scene captures a dynamic urban environment with skyscrapers, historic buildings, and busy pedestrians. Alex, wearing a helmet and casual work attire, navigates his bike with ease, reflecting a connection between technology and traditional city life. The morning sun illuminates the city, highlighting the energy of the start of a new day.
"""
text1_image_scene3 = """Scene 3: At Work
Alex is immersed in his work, surrounded by multiple monitors displaying code and software development tools. The office environment is lively, with colleagues collaborating in the background. Alex, focused and engaged, types away on his keyboard, solving complex problems. The scene conveys a sense of dedication and passion for technology, with notes and diagrams scattered around his workspace, and a cup of coffee nearby to keep him energized.
"""
text1_image_scene4 = """Scene 4: Evening Relaxation
Alex enjoys a peaceful moment in a lush, green park, sitting on a bench with a book in hand. The park is filled with tall trees, colorful flowers, and a small pond reflecting the sky. People are walking dogs and jogging in the background, creating a serene and lively atmosphere. Alex, taking a break from technology, is relaxed and content, absorbed in his novel. The setting sun casts a warm glow over the scene, enhancing the tranquility and beauty of the moment.
"""
text1_image_scene5 = """Scene 5: Nighttime Wind-down
Alex prepares for bed in his smart home, which is set to night mode with dim lights and ambient sounds. The scene shows a modern bedroom with a comfortable bed, smart gadgets around, and a window showing the night sky. Alex, in comfortable nightwear, is jotting down notes in a journal, reflecting on the day. The room exudes a calm and cozy atmosphere, with a book and a meditation mat nearby, highlighting his routine of winding down and gathering thoughts before sleep.
"""

# Categories +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Predefined categories and their respective scenes
categories = {
    "Custom GPT (5-Scenes)": ["text1_scene1", "text1_scene2", "text1_scene3", "text1_scene4", "text1_scene5"],
    "Gemini": ["text2_scene1", "text2_scene2", "text2_scene3", "text2_scene4", "text2_scene5"],
    "Mixtral Crazy": ["text3_scene1", "text3_scene2", "text3_scene3", "text3_scene4", "text3_scene5"],
    "Mixtral Normal": ["text4_scene1", "text4_scene2", "text4_scene3", "text4_scene4", "text4_scene5"],
    "Zephyr": ["text5_scene1", "text5_scene2", "text5_scene3", "text5_scene4", "text5_scene5"],
    "Phi-2": ["text6_scene1", "text6_scene2", "text6_scene3", "text6_scene4", "text6_scene5"]
}

def get_scene_text(scene_id):
    return globals()[scene_id]



# CrewAI +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def crewai_process_gemini(research_topic):
    # Define your agents with roles and goals
    GeminiAgent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
                   
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} create your story by writing at least one sentence about each bullet point 
        and make sure you have a transitional statement between scenes . BE VERBOSE.""",
        agent=GeminiAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[GeminiAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result



def crewai_process_mixtral_crazy(research_topic):
    # Define your agents with roles and goals
    MixtralCrazyAgent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_crazy      
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} create your story by writing at least one sentence about each bullet point 
        and make sure you have a transitional statement between scenes . BE VERBOSE.""",
        agent=MixtralCrazyAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[MixtralCrazyAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_mixtral_normal(research_topic):
    # Define your agents with roles and goals
    MixtralNormalAgent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_normal      
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} create your story by writing at least one sentence about each bullet point 
        and make sure you have a transitional statement between scenes . BE VERBOSE.""",
        agent=MixtralNormalAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[MixtralNormalAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_zephyr_normal(research_topic):
    # Define your agents with roles and goals
    ZephrNormalAgent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                ZephyrSearchTools.zephyr_normal     
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} create your story by writing at least one sentence about each bullet point 
        and make sure you have a transitional statement between scenes . BE VERBOSE.""",
        agent=ZephrNormalAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[ZephrNormalAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_phi2(research_topic):
    # Define your agents with roles and goals
    Phi2Agent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                Phi2SearchTools.phi2_search     
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} create your story by writing at least one sentence about each bullet point 
        and make sure you have a transitional statement between scenes . BE VERBOSE.""",
        agent=Phi2Agent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[Phi2Agent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result




# Initialize the HHEM model +++++++++++++++++++++++++++++++++++++++++++++++

    
model = CrossEncoder('vectara/hallucination_evaluation_model')

# Function to compute HHEM scores
def compute_hhem_scores(texts, summary):
    pairs = [[text, summary] for text in texts]
    scores = model.predict(pairs)
    return scores

# Define the Vectara query function
def vectara_query(query: str, config: dict):
    corpus_key = [{
        "customerId": config["customer_id"],
        "corpusId": config["corpus_id"],
        "lexicalInterpolationConfig": {"lambda": config.get("lambda_val", 0.5)},
    }]
    data = {
        "query": [{
            "query": query,
            "start": 0,
            "numResults": config.get("top_k", 10),
            "contextConfig": {
                "sentencesBefore": 2,
                "sentencesAfter": 2,
            },
            "corpusKey": corpus_key,
            "summary": [{
                "responseLang": "eng",
                "maxSummarizedResults": 5,
            }]
        }]
    }

    headers = {
        "x-api-key": config["api_key"],
        "customer-id": config["customer_id"],
        "Content-Type": "application/json",
    }
    response = requests.post(
        headers=headers,
        url="https://api.vectara.io/v1/query",
        data=json.dumps(data),
    )
    if response.status_code != 200:
        st.error(f"Query failed (code {response.status_code}, reason {response.reason}, details {response.text})")
        return [], ""

    result = response.json()
    responses = result["responseSet"][0]["response"]
    summary = result["responseSet"][0]["summary"][0]["text"]

    res = [[r['text'], r['score']] for r in responses]
    return res, summary


# Tabs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Create the main app with three tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Interpretive Number","5 Scene Writer", "Model Translator", "TruLens", "HHEM (hallucinations)", "Data & Graphs"])


with tab1:
    st.header("Human Interpretive Number (HIN)")
    
    st.write("")
    "How do LLMs interpret us and over “a period” of time what is the result of our interaction with them?"
    st.write("")
  
    image_comparison(
        img1="./data/robot.jpg",
        img2="./data/life.jpg",
        label1="What the LMM Sees (Simulated Life)",
        label2="What we See (Real Life)",
    )


with tab2:

    st.header("Five Scene Data")
    st.link_button("Create Five Scene Data", "https://chat.openai.com/g/g-17tElc18U-five-scene-writer")
    st.write("")
    "Average day in the life of a software engineer."
    st.write("")

    # Load your images (either from local files)
    image_paths = ['./data/text1_scene1.jpg', './data/text1_scene2.jpg', './data/text1_scene3.jpg', './data/text1_scene4.jpg', './data/text1_scene5.jpg']  # Updated image paths
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Define scenes for each image
    scenes = ['Scene 1: Morning', 'Scene 2: Commute', 'Scene 3: At Work', 'Scene 4: Evening', 'Scene 5: To Bed']  # Customize your scene labels
    
    # Create columns for the images
    cols = st.columns(5)  # Create 5 columns
    
    # Display images with scenes in their respective columns
    for col, image, scene in zip(cols, images, scenes):
        with col:
            st.image(image, use_column_width=True)  # Adjust image size to fit the column width
            st.write(scene)  # Display the scene label under the image

    
    st.text_area('5-Scenes:', text1 , height=400)
 

with tab3:
    st.header("Model Translator")

    # User input for the research topic
    #research_topic = st.text_area('Enter your research topic:', '', height=100)
    research_topic = text1_bullets


    # Selection box for the function to execute
    process_selection = st.selectbox(
        'Choose the process to run:',
        ('crewai_process_gemini', 'crewai_process_mixtral_crazy', 'crewai_process_mixtral_normal', 'crewai_process_zephyr_normal', 'crewai_process_phi2')
    )

    # Button to execute the chosen function
    if st.button('Run Process'):
        if research_topic:  # Ensure there's a topic provided
            if process_selection == 'crewai_process_gemini':
                result = crewai_process_gemini(research_topic)
            elif process_selection == 'crewai_process_mixtral_crazy':
                result = crewai_process_mixtral_crazy(research_topic)
            elif process_selection == 'crewai_process_mixtral_normal':
                result = crewai_process_mixtral_normal(research_topic)
            elif process_selection == 'crewai_process_zephyr_normal':
                result = crewai_process_zephyr_normal(research_topic)
            elif process_selection == 'crewai_process_phi2':
                result = crewai_process_phi2(research_topic)
            st.write(result)
        else:
            st.warning('Please enter a research topic.')
   

with tab4:
    st.header("TruLens")

    st.link_button("Cross Check Program to Confirm TruLens Results", "https://huggingface.co/spaces/eaglelandsonce/TruLens-TestSystem")

    texts = {
    "Custom GPT": text1,
    "Gemini": text2,
    "Mixtral Crazy": text3,
    "Mixtral Normal": text4,
    "Zephyr": text5,
    "Phi-2": text6,
    }

    # Initialize OpenAI client and create embeddings
    oai_client = OpenAI()
    oai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text1
    )


    selected_text_key = st.selectbox("Select a text document", options=list(texts.keys()), index=0)

    
    # Set up ChromaDB and embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=api_key,
                                                 model_name="text-embedding-ada-002")
    chroma_client = chromadb.Client()


    # Function to clear the vector store and add the selected text document
    def update_vector_store(chroma_client, vector_store_name, embedding_function, document_key, document_text):
    # Attempt to delete the existing collection if it exists
        try:
            chroma_client.delete_collection(name=vector_store_name)
        except Exception as e:
            print(f"Error clearing vector store: {e}")
        
        # Create or get the collection again
        vector_store = chroma_client.get_or_create_collection(name=vector_store_name, embedding_function=embedding_function)
        # Add the new document
        vector_store.add(document_key, documents=[document_text])
        return vector_store

    
    # vector_store = chroma_client.get_or_create_collection(name="Scenes", embedding_function=embedding_function)
    # vector_store.add("text1_info", documents=text1)


    # Update vector store based on selection
    vector_store_name = "Scenes"
    document_key = f"{selected_text_key}_info"
    selected_text = texts[selected_text_key]
    vector_store = update_vector_store(chroma_client, vector_store_name, embedding_function, document_key, selected_text)


    # Define RAG_from_scratch class
    class RAG_from_scratch:
        @instrument
        def retrieve(self, query: str) -> list:
            results = vector_store.query(
                query_texts=query,
                n_results=2
            )
            return results['documents'][0]
    
        @instrument
        def generate_completion(self, query: str, context_str: list) -> str:
            completion = oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "user", "content": 
                     f"We have provided context information below. \n"
                     f"---------------------\n"
                     f"{context_str}"
                     f"\n---------------------\n"
                     f"Given this information, please answer the question: {query}"}
                ]
            ).choices[0].message.content
            return completion
    
        @instrument
        def query(self, query: str) -> str:
            context_str = self.retrieve(query)
            completion = self.generate_completion(query, context_str)
            return completion
    
    rag = RAG_from_scratch()
   
    # Initialize feedback and evaluation mechanisms
    fopenai = fOpenAI()
    grounded = Groundedness(groundedness_provider=fopenai)
    
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(Select.RecordCalls.retrieve.rets.collect())
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    
    f_qa_relevance = (
        Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance")
        .on(Select.RecordCalls.retrieve.args.query)
        .on_output()
    )
    
    f_context_relevance = (
        Feedback(fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
        .on(Select.RecordCalls.retrieve.args.query)
        .on(Select.RecordCalls.retrieve.rets.collect())
        .aggregate(np.mean)
    )
    
    tru_rag = TruCustomApp(rag,
        app_id='RAG v1',
        feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])
    

  
    query = st.text_input("Ask a question about the Scenes")
    
    """
Scene 1 Morning:
How does Alex's technology-filled morning routine set the tone for his day as a software engineer?

Scene 2 Bike Ride:
What does Alex's commute reveal about his values and how he integrates technology into his daily life?

Scene 3 Office:
How do Alex's interactions with his team and coding work reflect the collaborative nature of software development?

Scene 4 Park:
How does Alex's evening relaxation activities influence his creativity and perspective on technology?

Scene 5 Home:
How do Alex's nighttime rituals contribute to his professional development and mental well-being?  
    """
    
    if st.button("Submit"):
    
        with st.spinner('Searching for information...'):
            with tru_rag as recording:
                answer = rag.query(query)
                final_tru = tru.get_leaderboard(app_ids=["RAG v1"])
            st.write(answer)
            st.write(final_tru)
            
            # Display feedback metrics (mockup, adjust based on your implementation)
            st.subheader("Feedback Metrics")
        
            records, feedback = tru.get_records_and_feedback(app_ids=["RAG v1"])
        
            st.write(records)
           
with tab5:

    st.header("HHEM-Vectara Hallucinations Measure (RAG)")
    

     # Category selection
    selected_category = st.selectbox("Select a Category", list(categories.keys()))
    
    # Scene selection based on the selected category
    selected_scene = st.selectbox("Select a Scene", categories[selected_category])
    
    # Use the selected scene to display its text in the query input field
    # Here, `get_scene_text(selected_scene)` dynamically fetches the text for the selected scene.
    query = st.text_area("Enter your text for query tuning", get_scene_text(selected_scene), height=150)

    voice_option = st.selectbox(
    'Choose a voice:',
    ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    )

    if st.button('Convert to Speech'):
        if query:
            try:
                response = oai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice_option,
                    input=query,
                )
                
                # Stream or save the response as needed
                # For demonstration, let's assume we save then provide a link for downloading
                audio_file_path = "output.mp3"
                response.stream_to_file(audio_file_path)
                
                # Display audio file to download
                st.audio(audio_file_path, format='audio/mp3')
                st.success("Conversion successful!")
                
                # Displaying the image with the same name as the selected scene
                image_file_path = f"./data/{selected_scene}.jpg"  # Adjust the directory as needed
                try:
                    st.image(image_file_path, caption=f"Scene: {selected_scene}")
                    """All images generated by Dall-E"""
                except Exception as e:
                    st.error(f"An error occurred while displaying the image: {e}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter some text to convert.")

    
    
    lambda_val = st.slider("Lambda Value", min_value=0.0, max_value=1.0, value=0.5)
    top_k = st.number_input("Top K Results", min_value=1, max_value=50, value=3)


    
    if st.button("Query Vectara"):
        config = {
    
            "api_key": os.environ.get("VECTARA_API_KEY", ""),
            "customer_id": os.environ.get("VECTARA_CUSTOMER_ID", ""),
            "corpus_id": os.environ.get("VECTARA_CORPUS_ID", ""),      
    
            "lambda_val": lambda_val,
            "top_k": top_k,
        }
    
        results, summary = vectara_query(query, config)
    
        if results:
            st.subheader("Summary")
            st.write(summary)
            
            st.subheader("Top Results")
            
            # Extract texts from results
            texts = [r[0] for r in results[:5]]
            
            # Compute HHEM scores
            scores = compute_hhem_scores(texts, summary)
            
            # Prepare and display the dataframe
            df = pd.DataFrame({'Fact': texts, 'HHEM Score': scores})
            st.dataframe(df)
        else:
            st.write("No results found.")


with tab6:
    
    st.header("Final Results HIN Number")
    st.write("HIN Score: Sum of Groundedness x HHEM Hullucination")
    st.write("HIN Score: Zephyr (40%) > Mixtral Normal (36%) > OpenAI (27%) >Mixtral Crazy (25%)> Gemini(16%)>Phi-2(14%)")
   

    
    st.write("")
    "Zephyr was the high performer with the Highest HIN Score."
    st.write("")

    # Load your images (either from local files)
    image_paths = ['./data/text5_scene1.jpg', './data/text5_scene2.jpg', './data/text5_scene3.jpg', './data/text5_scene4.jpg', './data/text5_scene5.jpg']  # Updated image paths
    images = [Image.open(image_path) for image_path in image_paths]
    
    # Define scenes for each image
    scenes = ['Scene 1: Morning', 'Scene 2: Commute', 'Scene 3: At Work', 'Scene 4: Evening', 'Scene 5: To Bed']  # Customize your scene labels
    
    # Create columns for the images
    cols = st.columns(5)  # Create 5 columns
    
    # Display images with scenes in their respective columns
    for col, image, scene in zip(cols, images, scenes):
        with col:
            st.image(image, use_column_width=True)  # Adjust image size to fit the column width
            st.write(scene)  # Display the scene label under the image

    st.write("HIN Model Plots")


    data = {
    "Scene 1": [0.48, 0.0575, 0.2482, 0.35, 0.4675, 0.0621],
    "Scene 2": [0.2208, 0.2508, 0.077, 0.3713, 0.5166, 0.0416],
    "Scene 3": [0.25, 0.18, 0.2745, 0.3942, 0.1342, 0.3640],
    "Scene 4": [0.2499, 0.1577, 0.2074, 0.203, 0.427, 0.0832],
    "Scene 5": [0.1248, 0.1343, 0.424, 0.4816, 0.494, 0.1064]
    }

    # Labels for the lines
    labels = ["OpenAI", "Gemini", "Mistral Crazy", "Mistral Normal", "Zephyr", "Phi-2"]
    
    # Create a DataFrame
    df = pd.DataFrame(data, index=labels)
    
    # Plotting with matplotlib
    plt.figure(figsize=(10, 5))
    for label in df.index:
        plt.plot(df.columns, df.loc[label, :], label=label)
    
    plt.legend()
    plt.xlabel('Scenes')
    plt.ylabel('Values')
    plt.title('Multiline Chart for Different Models')
    st.pyplot(plt)





    st.write("Raw Data")

    mydata=""" Note: Q means Scene:
    
OpenAI
Groundedness, Context Relevance, Answer Relevance, Hallucination
Q1: .50, .94, .97, .96
Q2: .46, .93, .97, .48
Q3: .50, .93, .95,.50
Q4: .51, .91, .95, .49
Q5:.48, .91, .95, .26
 
Hin Numbers 0.48, 0.2208, 0.25, 0.2499, 0.1248, with a total of approximately 1.3255.
 
Hin Sum = 26.6%
 
Gemini
Groundedness, Context Relevance, Answer Relevance, Hallucination
 
Q1: .25, .9, .9, .23
Q2: .33, .9, .93, .33
Q3: .2, .88, .93, .2
Q4: .19, .87, .93, .83
Q5: .17, .88, .93, .79
 
Hin Numbers 0.0575, 0.2508, 0.18, 0.1577, 0.1343, with a total of approximately 0.7803.
Hin Sum = 16.6%
 
 
Mixtral Crazy
Groundedness, Context Relevance, Answer Relevance, Hallucination
Q1: .73, .85, .9, .34
Q2: .77, .88, .9, .1
Q3: .61, .88, .92, .45
Q4: .61, .87, .93, .34
Q5:  .53, .87, .93, .8
 
Hin Numbers 0.2482, 0.077, 0.2745, 0.2074, 0.424, with a total of approximately 1.2311.
HIN Sum = 24.6%
 
Mixtral Normal
Groundedness, Context Relevance, Answer Relevance, Hallucination
 
Q1: .7, .9, 1, .5
Q2: .79, .9, 1, .47
Q3: .73, .9, .97, .54
Q4: .7, .88, .98, .29
Q5:  .56, .88, .96, .86
 
Hin Numbers 0.35, 0.3713, 0.3942, 0.203, 0.4816, with a total of approximately 1.8001
 
HIN Sum = 36%
 
Zephyr
Groundedness, Context Relevance, Answer Relevance, Hallucination
 
Q1: .85, .9, .97, .55
Q2: .63, .9, .98, .82
Q3: .61, .9, .96, .22
Q4: .61, .88, .95, .70
Q5: .52, .89, .94, .95
 
Hin Numbers 0.4675, 0.5166, 0.1342, 0.427, 0.494, with a total of approximately 2.0393. 
 
HIN Sum = 40%
 
Phi-2
Groundedness, Context Relevance, Answer Relevance, Hallucination
 
Q1:  .23, .9, .9, .27
Q2: .32, .9, .9, .13
Q3: .40, .74, .9, .91
Q4: .32, .65, .9, .26
Q5: .28, .69, .91,.38
 
Hin Numbers 0.0621, 0.0416, 0.3640, 0.0832, 0.1064, with a total of approximately 0.6573. 
 
HIN Sum =13.6%
 
 
Zephyr (40%) > Mixtral Normal (36%) > OpenAI (27%) >Mixtral Crazy (25%)> Gemini(16%)>Phi-2(14%)

    
    
    
    """

    
    st.text_area('5-Scenes:', mydata , height=400)
   