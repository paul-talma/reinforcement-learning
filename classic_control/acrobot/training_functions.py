from replay_buffer import ExperienceReplayBuffer, Transition
import torch
import torch.nn as nn


def soft_update(policy_net, target_net, tau):
    """
    Updates the parameters of the target using parameters of policy net.
    Biased toward old values by (1 - tau)

    Params:
    policy_net: deep Q-network
    target_net: deep Q-network
    tau: weight assigned to policy_net parameter values
    """
    policy_net_state_dict = policy_net.state_dict()
    target_net_state_dict = target_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)


def optimize(
    policy_net,
    target_net,
    buffer: ExperienceReplayBuffer,
    batch_size,
    device,
    discount_factor,
    optimizer,
):
    if len(buffer) < batch_size:
        return
    transitions = buffer.sample_experience(batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_vals = policy_net(state_batch).gather(1, action_batch)

    next_state_vals = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_vals[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )

    # compute expected Q-values
    expected_q_vals = discount_factor * next_state_vals + reward_batch

    # compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_vals, expected_q_vals.unsqueeze(1))

    # optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
