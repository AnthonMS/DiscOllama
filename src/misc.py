async def react_thinking(message, user=False):
    if not user:
        await message.add_reaction('🤔')
    else:
        await message.remove_reaction('🤔', user)

async def react_poop(message, user=False):
    if not user:
        await message.add_reaction('💩')
    else:
        await message.remove_reaction('💩', user)
        
async def react_no(message, user=False):
    if not user:
        await message.add_reaction('❌')
    else:
        await message.remove_reaction('❌', user)
        
async def react_ok(message, user=False):
    if not user:
        await message.add_reaction('👌')
    else:
        await message.remove_reaction('👌', user)
        
async def react_test(message, user=False):
    if not user:
        await message.add_reaction('🚫')
    else:
        await message.remove_reaction('🚫', user)
        
# 🙅‍♂️



async def get_model_list(ollama):
    model_list = await ollama.list()
    available_models = []
    for model in model_list['models']:
        available_models.append(f"{model['name']}")
    return available_models