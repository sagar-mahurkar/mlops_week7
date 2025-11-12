# Week 7- Graded Assignment 7

### Assignment Objective

Building on last week's CI/CD pipeline

This week, we will be scaling the homework IRIS classification pipeline to handle multiple concurrent inferences and observe bottlenecks.

1. Extend your existing Github CI/CD workflow to stress test the deployment

2. Use wrk to stimulate the scenario of high number(>1000) of requests after successful deployment

3. Demonstrate Kubernetes auto scaling with max_pods : 3 and default pod availability of 1

4. Observe bottlenecks when auto scaling is restricted to 1 pod and request concurrency increased from 1000 to 2000

