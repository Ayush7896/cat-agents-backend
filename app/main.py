from app.graph import build_workflow
from fastapi import FastAPI, HTTPException
from app.models.schemas import CATRequest, CATResponse
from fastapi.middleware.cors import CORSMiddleware



workflow = build_workflow()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model = CATResponse)
async def ask(request: CATRequest):
    try:
        # passage = request.passage
        # user_query = request.user_query
        thread_id = request.thread_id

        config = {"configurable": {"thread_id": thread_id}}

        # initial_state = {
        #     "passage":passage,
        #     "user_query": user_query
        # }
        user_input_state = {"user_query": request.user_query, "passage": request.passage}
        final_state = workflow.invoke(user_input_state, config = config)
        # print(f"persistence checking  {final_state.keys()}")
        # print(f"workflow states {workflow.get_state(config)}")
        final_answer = final_state.get("final_answer", "No answer generated")
        return CATResponse(final_answer=final_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

    
# if __name__ == "__main__":
#     workflow = build_workflow()
#     initial_state = {
#         "passage": """There is a group in the space community who view the solar system not as an opportunity to expand human potential but as a nature preserve, forever the provenance of an elite group of scientists and their sanitary robotic probes. These planetary protection advocates [call] for avoiding "harmful contamination" of celestial bodies. Under this regime, NASA incurs great expense sterilizing robotic probes in order to prevent the contamination of entirely theoretical biospheres. . . .

#         Transporting bacteria would matter if Mars were the vital world once imagined by astronomers who mistook optical illusions for canals. Nobody wants to expose Martians to measles, but sadly, robotic exploration reveals a bleak, rusted landscape, lacking oxygen and flooded with radiation ready to sterilize any Earthly microbes. Simple life might exist underground, or down at the bottom of a deep canyon, but it has been very hard to find with robots. . . . The upsides from human exploration and development of Mars clearly outweigh the welfare of purely speculative Martian fungi. . . .

#         The other likely targets of human exploration, development, and settlement, our moon and the asteroids, exist in a desiccated, radiation-soaked realm of hard vacuum and extreme temperature variations that would kill nearly anything. It's also important to note that many international competitors will ignore the demands of these protection extremists in any case. For example, China recently sent a terrarium to the moon and germinated a plant seed—with, unsurprisingly, no protest from its own scientific community. In contrast, when it was recently revealed that a researcher had surreptitiously smuggled super-resilient microscopic tardigrades aboard the ill-fated Israeli Beresheet lunar probe, a firestorm was unleashed within the space community. . . .

#         NASA's previous human exploration efforts made no serious attempt at sterility, with little notice. As the Mars expert Robert Zubrin noted in the National Review, U.S. lunar landings did not leave the campsites cleaner than they found it. Apollo's bacteria-infested litter included bags of feces. Forcing NASA's proposed Mars exploration to do better, scrubbing everything and hauling out all the trash, would destroy NASA's human exploration budget and encroach on the agency's other directorates, too. Getting future astronauts off Mars is enough of a challenge, without trying to tote weeks of waste along as well.

#         A reasonable compromise is to continue on the course laid out by the U.S. government and the National Research Council, which proposed a system of zones on Mars, some for science only, some for habitation, and some for resource exploitation. This approach minimizes contamination, maximizes scientific exploration . . . Mars presents a stark choice of diverging human futures. We can turn inward, pursuing ever more limited futures while we await whichever natural or manmade disaster will eradicate our species and life on Earth. Alternatively, we can choose to propel our biosphere further into the solar system, simultaneously protecting our home planet and providing a backup plan for the only life we know exists in the universe. Are the lives on Earth worth less than some hypothetical microbe lurking under Martian rocks?
#         """,

#         "user_query":""" for above passage, how to answer inference, strengthen the argument, weaken the argument questions and what questions can be asked to achieve this?
#                 """
#     }
#     final_state = workflow.invoke(initial_state)
#     print(final_state['final_answer'])
