import openai
openai.api_base = "http://10.1.114.10:8091/v1"
openai.api_key = "none"

# 使用流式回复的请求
# for chunk in openai.ChatCompletion.create(
#     model="Qwen",
#     messages=[
#         {"role": "user", "content": "你好"}
#     ],
#     stream=True
#     # 流式输出的自定义stopwords功能尚未支持，正在开发中
# ):
#     if hasattr(chunk.choices[0].delta, "content"):
#         print(chunk.choices[0].delta.content, end="", flush=True)
#relas = get_2hop_relations('m.0b787yg')
#print(relas)

# 不使用流式回复的请求
prompt = """

Question: which plant disease is hosted on core eudicots?
Logical Form: (AND biology.plant_disease (JOIN biology.plant_disease.plant_disease_triangle (JOIN biology.plant_disease_triangle.host core eudicots)))
Question: negative regulation of epinephrine secretion is the narrower group of which gene ontology group?
Logical Form: (AND biology.gene_ontology_group (JOIN biology.gene_ontology_group.narrower_group negative regulation of epinephrine secretion))
Question: which collection activity has the type of thing collected that is the parent category of my child?
Logical Form: (AND interests.collection_activity (JOIN interests.collection_activity.type_of_thing_collected (JOIN (R interests.collection_category.parent_category) My Child)))
Question: brad mays made how many quotes?
Logical Form: (COUNT (AND media_common.quotation (JOIN (R people.person.quotations) Brad Mays)))
Question: who was the constitutional convention delegate who used the quotations the things which hurt, instruct.?
Logical Form: (AND law.constitutional_convention_delegate (JOIN people.person.quotations The things which hurt, instruct.))
Question: the fictional calendar system preceded by first age is a feature of which fictional universe?
Logical Form: (AND fictional_universe.fictional_universe (JOIN (R fictional_universe.fictional_calendar_system.used_in_fictional_universes) (JOIN fictional_universe.fictional_calendar_system.preceded_by First Age)))
Question: what digital camera has a tele focal length of 270.0?
Logical Form: (AND digicams.digital_camera (JOIN digicams.digital_camera.tele_focal_length 270.0^^http://www.w3.org/2001/XMLSchema#float))
Question: what basketball coach has 9 career wins recorded?
Logical Form: (AND basketball.basketball_coach (JOIN basketball.basketball_coach.season_wins 9^^http://www.w3.org/2001/XMLSchema#integer))
Question: what chess play got married after nov. the 24th, 1924
Logical Form: (AND chess.chess_player (JOIN people.person.spouse_s (ge people.marriage.from 1924-11-24^^http://www.w3.org/2001/XMLSchema#date)))
Question: the disney ride with the biggest drop has what amusement ride theme?
Logical Form: (ARGMAX amusement_parks.ride_theme (JOIN amusement_parks.ride_theme.rides amusement_parks.ride.drop))
Question: tulane green wave men's basketball is coached by who?
Logical Form: (AND basketball.basketball_coach (JOIN basketball.basketball_coach.team Tulane Green Wave men's basketball))
Question: how many quotations are there about liberty or death?
Logical Form: (COUNT (AND media_common.quotation (JOIN media_common.quotation.subjects (JOIN media_common.quotation_subject.quotations_about_this_subject liberty or death))))
Question: which cloud has a maximum altitude of 1500.0?
Logical Form: (AND meteorology.cloud (JOIN meteorology.cloud.maximum_altitude_m 1500.0^^http://www.w3.org/2001/XMLSchema#float))
Question: what's the type of medical trials sponsored by forest laboratories?
Logical Form: (AND medicine.medical_trial_type (JOIN medicine.medical_trial_type.medical_trials (JOIN medicine.medical_trial.sponsor forest laboratories)))
Question: what brand is the product search engine optimization?
Logical Form: (AND business.brand (JOIN business.brand.products Search Engine Optimization))
Question: the longest release track of howe recordings is what?
Logical Form: (ARGMAX (AND music.release_track (JOIN (R music.recording.tracks) (JOIN (R music.engineer.tracks_engineered) howe))) music.release_track.length)
Question: billie jean has what release track?
Logical Form: (AND music.release_track (JOIN music.release_track.release Billie Jean))
Question: the 5th power and more mission: impossible belong to which genre of music?
Logical Form: (AND music.genre (AND (JOIN (R music.album.genre) More Mission: Impossible) (JOIN (R music.album.genre) The 5th Power)))
Question: mosley v news group newspapers ltd are the cases of what judge?
Logical Form: (AND law.judge (JOIN law.judge.cases Mosley v News Group Newspapers Ltd))
Question: cyborg was a character of which comic book story?
Logical Form: (AND comic_books.comic_book_story (JOIN comic_books.comic_book_story.characters Cyborg))
Question: r. brooke jackson presided over which legal cases?
Logical Form: (AND law.legal_case (JOIN law.legal_case.judges R. Brooke Jackson))
Question: what is the mission destination of new horizons?
Logical Form: (AND spaceflight.mission_destination (JOIN spaceflight.mission_destination.missions_sent_here New Horizons))
Question: what musical genre has a subgenres that has a parent genres of jump-up?
Logical Form: (AND music.genre (JOIN music.genre.subgenre (JOIN music.genre.parent_genre Jump-up)))
Question: what is the name of the architect who is known for postmodern architecture?
Logical Form: (AND architecture.architect (JOIN architecture.architect.architectural_style Postmodern architecture))
Question: what is the family of hilborne roosevelt?
Logical Form: (AND people.family (JOIN people.family.members Hilborne Roosevelt))
Question: the c.a.f.e. informatique et telecommunications is registered under what domain?
Logical Form: (AND internet.top_level_domain (JOIN internet.top_level_domain.registry C.A.F.E. Informatique et Telecommunications))
I provide 40 exapmles of question and its logical form.  You need to generate the logical form just for the following question. Your output format should be "Logical Form: " and the logical form of the question, no more other words. Remeber, you should just generate the logical form for the following question.
Question: what is the role of opera designer gig who designed the telephone / the medium?
Logical Form:
"""
prompt_1="""
Question: the variants of harvard architecture is manufactured by which processor manufacturer?
Type of the question: Composition
Question: blood disorder are what types of diseases or medical conditions?
Type of the question: Composition
Question: what editions are in the series of library of the mystic arts?
Type of the question: Composition
Question: french republic participated in which of the multi-event tournaments?
Type of the question: Comparison
Question: what is a restaurant whos menu includes mediterranean cuisine and seafood?
Type of the question: Composition
Question: who is the most overweight guitarist?
Type of the question: Comparison
Question: what is the focus of the project that includes design and construction of the millau viaduct?
Type of the question: Comparison
Question: planet ceres is the destination of what space mission?
Type of the question: 
 """
prompt_2="""
 Question: op\u00e9ra bastille is the premiere production of what opera?\nLogical form candidates: \n(AND architecture.building_function (JOIN architecture.building_function.buildings Op\u00e9ra Bastille))\n(COUNT (AND architecture.building_function (JOIN architecture.building_function.buildings Op\u00e9ra Bastille)))\n(AND opera.opera (JOIN opera.opera.premiere_production Op\u00e9ra Bastille))\n(COUNT (AND opera.opera (JOIN opera.opera.premiere_production Op\u00e9ra Bastille)))\n(AND location.my_district (JOIN location.location.contains Op\u00e9ra Bastille))\nTrue logical form: (AND opera.opera (JOIN opera.opera.premiere_production Op\u00e9ra Bastille))\n----------------------------------------\nQuestion: what opera production was produced by david pountney?\nLogical form candidates: \n(AND education.education (JOIN education.education.student David Pountney))\n(COUNT (AND education.education (JOIN education.education.student David Pountney)))\n(AND opera.opera_production (JOIN opera.opera_production.producer David Pountney))\n(COUNT (AND opera.opera_production (JOIN opera.opera_production.producer David Pountney)))\n(AND music.concert_film (JOIN film.film.directed_by David Pountney))\nTrue logical form: (AND opera.opera_production (JOIN opera.opera_production.producer David Pountney))\n----------------------------------------\nQuestion: which is the role of opera designer gig who designed peter grimes?\nLogical form candidates: \n(AND award.award_nomination (JOIN award.award_nomination.nominated_for Peter Grimes))\n(COUNT (AND award.award_nomination (JOIN award.award_nomination.nominated_for Peter Grimes)))\n(AND opera.opera_role (JOIN opera.opera_role.opera Peter Grimes))\n(COUNT (AND opera.opera_role (JOIN opera.opera_role.opera Peter Grimes)))\n(AND opera.opera (JOIN opera.opera.productions Peter Grimes))\nTrue logical form: (AND opera.opera_designer_role (JOIN (R opera.opera_designer_gig.design_role) (JOIN (R opera.opera_production.designers) Peter Grimes)))\n----------------------------------------\nQuestion: identify the musical voice that is used as character voice with basso cantante in jacopo ferretti's opera?\nLogical form candidates: \n(AND opera.opera_character_voice (JOIN opera.opera_character_voice.voice Bass))\n(COUNT (AND opera.opera_character_voice (JOIN opera.opera_character_voice.voice Bass)))\n(AND music.group_member (JOIN music.group_member.vocal_range Bass))\n(COUNT (AND music.group_member (JOIN music.group_member.vocal_range Bass)))\n(AND music.group_membership (JOIN music.group_membership.role Bass))\nTrue logical form: (AND music.voice (JOIN (R opera.opera_character_voice.voice) (AND (JOIN opera.opera_character_voice.voice Bass) (JOIN (R opera.opera.characters) (JOIN opera.opera.librettist Jacopo Ferretti)))))\n----------------------------------------\nQuestion: which opera production was performed at the mittels\u00e4chsisches theater?\nLogical form candidates: \n(AND opera.opera_production (JOIN opera.opera_production.performed_at Mittels\u00e4chsisches Theater))\n(COUNT (AND opera.opera_production (JOIN opera.opera_production.performed_at Mittels\u00e4chsisches Theater)))\n(AND opera.opera_production (JOIN opera.opera_production.producing_company Mittels\u00e4chsisches Theater))\n(COUNT (AND opera.opera_production (JOIN opera.opera_production.producing_company Mittels\u00e4chsisches Theater)))\n(AND opera.opera_production_venue_relationship (JOIN opera.opera_production_venue_relationship.opera_house Mittels\u00e4chsisches Theater))\nTrue logical form: (AND opera.opera_production (JOIN opera.opera_production.performed_at Mittels\u00e4chsisches Theater))\n----------------------------------------\n\n    I provide you 5 examples question with its logical form candidates and true logical form. And please generate the correct logical form for the following question according to the context and logical form candidates. Attention, I  would like you to generate the true logical form on your own rather than simply selecting from candidates. The candidates for each question are only for reference. It means that the true logical form maybe not in the candidates. You only need to output the true logical form for the following question. It means that you needn't output the question and other words. And if thers is no candidate for the question, you should generate the correct logical form based on your knowledge. Besides,quotation marks are not allowed in your response.Your ouput format should be \"The true logical form:\" and the logical form. No other words.You should not output ant Explanation. \n    ----------------------------------------\nQuestion: what is the role of opera designer gig who designed the telephone / the medium?\nLogical form candidates: \n(AND opera.opera_designer_gig (JOIN opera.opera_designer_gig.opera The Telephone / The Medium))\n(COUNT (AND opera.opera_designer_gig (JOIN opera.opera_designer_gig.opera The Telephone / The Medium)))\n(AND opera.opera_producer (JOIN opera.opera_producer.operas_produced The Telephone / The Medium))\n(COUNT (AND opera.opera_producer (JOIN opera.opera_producer.operas_produced The Telephone / The Medium)))\n(AND award.award_honor (JOIN award.award_honor.honored_for The Telephone / The Medium))\nTrue logical form: \n----------------------------------------\
    """
#  The words like " Sure! Here's the type of question you provided" should not appear in your output.
prompt_3 = """
  Give the correct logical form which can be executed on the knowledge graph to answer the question.\noxybutynin chloride 5 extended release film coated tablet is the ingredients of what routed drug?

"""
"""
response = openai.Completion.create(
    model="meta-llama/Llama-2-13B-hf",
    prompt=prompt,
    stream=False,
    stop=[] # 在此处添加自定义的stop words 例如ReAct prompting时需要增加： stop=["Observation:"]。
)
"""
prompt_test = """
Generate logical form for the following question. Your output format should be "Logical Form: " and the logical form of the question, no more other words. Remeber, you should just generate the logical form for last question.

Question: which plant disease is hosted on core eudicots?
Logical Form: (AND biology.plant_disease (JOIN biology.plant_disease.plant_disease_triangle (JOIN biology.plant_disease_triangle.host core eudicots)))

Question: negative regulation of epinephrine secretion is the narrower group of which gene ontology group?
Logical Form: (AND biology.gene_ontology_group (JOIN biology.gene_ontology_group.narrower_group negative regulation of epinephrine secretion))

Question: which collection activity has the type of thing collected that is the parent category of my child?
Logical Form: (AND interests.collection_activity (JOIN interests.collection_activity.type_of_thing_collected (JOIN (R interests.collection_category.parent_category) My Child)))

Question: brad mays made how many quotes?
Logical Form: (COUNT (AND media_common.quotation (JOIN (R people.person.quotations) Brad Mays)))

Question: who was the constitutional convention delegate who used the quotations the things which hurt, instruct.?
Logical Form: (AND law.constitutional_convention_delegate (JOIN people.person.quotations The things which hurt, instruct.))

Question: the fictional calendar system preceded by first age is a feature of which fictional universe?
Logical Form: (AND fictional_universe.fictional_universe (JOIN (R fictional_universe.fictional_calendar_system.used_in_fictional_universes) (JOIN fictional_universe.fictional_calendar_system.preceded_by First Age)))

Question: what digital camera has a tele focal length of 270.0?
Logical Form: (AND digicams.digital_camera (JOIN digicams.digital_camera.tele_focal_length 270.0^^http://www.w3.org/2001/XMLSchema#float))

Question: what basketball coach has 9 career wins recorded?
Logical Form: (AND basketball.basketball_coach (JOIN basketball.basketball_coach.season_wins 9^^http://www.w3.org/2001/XMLSchema#integer))

Question: what chess play got married after nov. the 24th, 1924
Logical Form: (AND chess.chess_player (JOIN people.person.spouse_s (ge people.marriage.from 1924-11-24^^http://www.w3.org/2001/XMLSchema#date)))

Question: the disney ride with the biggest drop has what amusement ride theme?
Logical Form: (ARGMAX amusement_parks.ride_theme (JOIN amusement_parks.ride_theme.rides amusement_parks.ride.drop))

Question: tulane green wave men's basketball is coached by who?
Logical Form: (AND basketball.basketball_coach (JOIN basketball.basketball_coach.team Tulane Green Wave men's basketball))

Question: how many quotations are there about liberty or death?
Logical Form: (COUNT (AND media_common.quotation (JOIN media_common.quotation.subjects (JOIN media_common.quotation_subject.quotations_about_this_subject liberty or death))))

Question: which cloud has a maximum altitude of 1500.0?
Logical Form: (AND meteorology.cloud (JOIN meteorology.cloud.maximum_altitude_m 1500.0^^http://www.w3.org/2001/XMLSchema#float))

Question: what's the type of medical trials sponsored by forest laboratories?
Logical Form: (AND medicine.medical_trial_type (JOIN medicine.medical_trial_type.medical_trials (JOIN medicine.medical_trial.sponsor forest laboratories)))

Question: what brand is the product search engine optimization?
Logical Form: (AND business.brand (JOIN business.brand.products Search Engine Optimization))

Question: the longest release track of howe recordings is what?
Logical Form: (ARGMAX (AND music.release_track (JOIN (R music.recording.tracks) (JOIN (R music.engineer.tracks_engineered) howe))) music.release_track.length)

Question: billie jean has what release track?
Logical Form: (AND music.release_track (JOIN music.release_track.release Billie Jean))

Question: the 5th power and more mission: impossible belong to which genre of music?
Logical Form: (AND music.genre (AND (JOIN (R music.album.genre) More Mission: Impossible) (JOIN (R music.album.genre) The 5th Power)))

Question: mosley v news group newspapers ltd are the cases of what judge?
Logical Form: (AND law.judge (JOIN law.judge.cases Mosley v News Group Newspapers Ltd))

Question: cyborg was a character of which comic book story?
Logical Form: (AND comic_books.comic_book_story (JOIN comic_books.comic_book_story.characters Cyborg))

Question: r. brooke jackson presided over which legal cases?
Logical Form: (AND law.legal_case (JOIN law.legal_case.judges R. Brooke Jackson))

Question: what is the mission destination of new horizons?
Logical Form: (AND spaceflight.mission_destination (JOIN spaceflight.mission_destination.missions_sent_here New Horizons))

Question: what musical genre has a subgenres that has a parent genres of jump-up?
Logical Form: (AND music.genre (JOIN music.genre.subgenre (JOIN music.genre.parent_genre Jump-up)))

Question: what is the name of the architect who is known for postmodern architecture?
Logical Form: (AND architecture.architect (JOIN architecture.architect.architectural_style Postmodern architecture))

Question: what is the family of hilborne roosevelt?
Logical Form: (AND people.family (JOIN people.family.members Hilborne Roosevelt))

Question: the c.a.f.e. informatique et telecommunications is registered under what domain?
Logical Form: (AND internet.top_level_domain (JOIN internet.top_level_domain.registry C.A.F.E. Informatique et Telecommunications))

Question: what is the role of opera designer gig who designed the telephone / the medium?
Logical Form:
"""
response = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-13B-hf",
    messages=[{"role":"user","content":"hello"}]
    ,stream=False)
#print(response)
print(response.choices[0].message.content)
#print(response["choices"][0]['text'].strip())
